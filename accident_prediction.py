"""
@author: Tejas Arya (ta2763)
@author: Amritha Venkataramana (axv3602)

"""
import numpy
import os
import glob
import cv2
from keras.utils import to_categorical
import sklearn
import sklearn.metrics
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras.backend.tensorflow_backend as tfback

"""
Class AccidentPrediction classifies accident videos as containing accident (1) or no accident (0). The data is generated 
from videos in appropriate directories (Training or Testing). Each directory have positive or negative videos, where
positive video implies an accident video and negative video implies a non-accident video.
"""


class AccidentPrediction:

    def __init__(self):
        """
        init method is used to initialize global variables.
        """
        # train data path of positive videos
        self.training_path_positive = 'dataset_labelled/features/training/positive'
        # train data path of negative videos
        self.training_path_negative = 'dataset_labelled/features/training/negative'
        # test data path of positive videos
        self.testing_path_positive = 'dataset_labelled/features/testing/positive'
        # test data path of negative videos
        self.testing_path_negative = 'dataset_labelled/features/testing/negative'
        # trial data path
        self.trial = 'dataset/features/trial'
        # trial out path
        self.trial_out = 'dataset/features/trial/out'
        # batch size for data
        self.batch_size = 15
        # number of classes (Accident (1) or Non-Accident (0))
        self.num_classes = 2
        # number of hidden rows
        self.row_hidden = 128
        # number of hidden columns
        self.col_hidden = 128
        # number of frames per video, number of rows in a frame, number of columns in a frame
        self.frame, self.row, self.col = (98, 200, 200)
        # number of epochs to train the model on
        self.epochs = 30
        # classes
        self.classes = [0, 1]



    def make_frames_helper(self, file):
        """
        method to convert video to frames to make it passable to the network
        :param file:
        :return: frames
        """
        # read video file
        vidcap = cv2.VideoCapture(file)
        # intialize file reader
        success, img = vidcap.read()
        error = ''
        # intialize success as True to know there is a video frame
        success = True
        # while there is a video frame
        while success:
            # read successive video frames till there are more video frames
            success, img = vidcap.read()
            # initialize a frame array to store frames in a video
            frames = []
            # for 99 frames
            for j in range(0,99):
                try:
                    # read video frames for 99 frames
                    success, img = vidcap.read()
                    # convert video frames to grayscale
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # downscale to 200x200 size for ease of computation
                    tmp = cv2.resize(gray, (200, 200))
                    # append video frame to frames array
                    frames.append(tmp)
                except:
                    pass
            # ensuring frame size is correct
            if numpy.shape(frames[0]) != (200, 200):
                error = 'Video is not the correct resolution.'
        # release video object from vidcap
        vidcap.release()
        # return video frame
        return frames

    def make_frames_helper_labelled(self,path):
        frames = []
        for files in os.listdir(path):
            frames.append(cv2.imread(files))
        return frames

    def make_frames(self, dir_name):
        """
        method for design test to output frames
        :param dir_name:
        """
        # for files in given directory
        for files in os.listdir(dir_name):
            # if it is a video file, create frames
            if files[-4:] == '.mp4':
                path = dir_name + '/' + files
                all_frames, error = self.make_frames_helper(path)
                count = 1
                new_dir = dir_name + "/out/" + files[:-4]
                os.mkdir(new_dir)
                # store frames in a out folder
                for frames in all_frames:
                    new_path = dir_name + "/out/" + files[:-4] + "/" + str(count) + '.jpeg'
                    count += 1
                    cv2.imwrite(new_path, frames)

    def combine_train_data(self, dir_name):
        """

        :param dir_name:
        :return: all file names and their corresponding labels
        """
        # container for names of positive files
        pos = glob.glob(dir_name + '/positive/*')
        # container for names of negative files
        neg = glob.glob(dir_name + '/negative/*')
        # combine containers for positive and negative files in the given order
        all_files = numpy.concatenate((pos, neg))
        # create labels from type of file (positice or negative) for their respective amount of videos
        labels = numpy.concatenate(([1] * len(pos), [0] * len(neg)))
        # return all file names and labels
        return all_files, labels
        # self.create_labels(labels)

    def create_labels(self, values):
        """
        method for design test to create one-hot encoding
        :param values:
        """
        n_values = numpy.max(values) + 1
        print(numpy.eye(n_values)[values])

    def create_data_for_training(self, data_file_name):
        """
        from names of data files, generate data to train the model or test the trained model
        :param data_file_name:
        :return: total data
        """
        # initialize an array to store frames of all videos
        total_data = [0] * len(data_file_name)
        # for video paths in file names, get frames for the video using make_frames_helper method and store it in array
        # total_data
        for i, video_path in enumerate(data_file_name):
            t = self.make_frames_helper(video_path)
            total_data[i] = t
            # if numpy.shape(t) == (99, 144, 256):  ### double check to make sure the shape is correct, and accept
            #     seq1[i] = t
        # return total_data
        return total_data

    def network(self):
        """
        define the model, its input type, input layers, prediction layer, loss functions, optimizer and metrics for
        evaluation
        :return: model
        """
        # define input shape --> number of inputs, number of rows and columns
        x = Input(shape=(self.frame, self.row, self.col))
        # rows passed through TimeDistributed LSTM returning a vector of 128 pixels
        encoded_rows = TimeDistributed(LSTM(self.row_hidden))(x)
        # corresponding columns passed through LSTM. Input is encoded rows from previous layers
        encoded_columns = LSTM(self.col_hidden)(encoded_rows)
        # print(encoded_columns)
        # use length of test data instead of num_classes
        # prediction layer containing two possible outputs with activation function 'softmax'
        prediction = Dense(self.num_classes, activation='softmax')(encoded_columns)
        # combining the model of previous layers and input format
        model = Model(x, prediction)
        # compile the model with loss categorical_crossentropy throughout the model, using NAdam optimizer and accuracy
        # as evaluation metric
        model.compile(loss='categorical_crossentropy',
                      optimizer='NAdam',
                      metrics=['accuracy'])
        # return the model
        return model

    def train_model(self, model, data, labels):
        batch_size = 2
        x = 0;
        # initialize a file path to store the best model along side training when model is learning from training data
        filepath = 'models.hdf5'
        # tell the model checkpoint and where to store the best model
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
        # store the model save point in callbacks_list to be passed while fitting the model
        callbacks_list = [checkpoint]
        # combine data and labels together
        c = list(zip(data, labels))
        # randomly shuffle the combined labels and corresponding and data ensuring they end up at correct place after
        # shuffling
        numpy.random.shuffle(c)
        # break the combined list so that they end up at correct location in their respective lists
        shuffled_data, shuffled_labels = zip(*c)
        # store data in a numpy array
        shuffled_data = numpy.asarray(shuffled_data)
        # store labels in a numpy array
        shuffled_labels = numpy.asarray(shuffled_labels)
        # store the data in batches of batch_size 15
        data_batch = [shuffled_data[i:i + self.batch_size] for i in range(0, len(shuffled_data), self.batch_size)]
        print("LENGTH DATA BATCH", len(data_batch))
        # store tha labels in batches of batch size 15
        label_batch = [shuffled_labels[i:i + self.batch_size] for i in range(0, len(shuffled_labels), self.batch_size)]
        # print('------')
        # initialize empty containers for storing the data from respective batch in the iteration
        dataX = [None]
        labelX = [None]
        # for batches of 15 videos each, train the model, for 5 epochs each
        for x in range(len(data_batch)):
            # take xth batch data
            dataX = data_batch[x]
            print("BATCH No: ",x)
            # xth batch labels
            labelX = label_batch[x]
            # convert labels to one hot encoding
            labelX = to_categorical(labelX)
            # print(numpy.shape(dataX))
            # print(numpy.shape(labelX))
            # fit the model with current batch data and labels for 5 epochs each and store the best model along the way
            model.fit(dataX, labelX, batch_size=len(dataX), epochs=5, callbacks=callbacks_list)
        # store the final best model
        model.save('finalmodel.h5')
        # return the model, data and label for visualization purposes
        return model, dataX, labelX

    def test_model(self, model, data, labels):
        # combine data and labels
        c = list(zip(data, labels))
        # randomly shuffle the combined labels and corresponding and data ensuring they end up at correct place after
        # shuffling
        numpy.random.shuffle(c)
        # break the combined list so that they end up at correct location in their respective lists
        shuffled_data, shuffled_labels = zip(*c)
        # store data in a numpy array
        shuffled_data = numpy.asarray(shuffled_data)
        # store labels in a numpy array
        shuffled_labels = numpy.asarray(shuffled_labels)
        # store the data in batches of 15
        data_batch = [shuffled_data[i:i + self.batch_size] for i in range(0, len(shuffled_data), self.batch_size)]
        # store the labels in batches of 15
        label_batch = [shuffled_labels[i:i + self.batch_size] for i in range(0, len(shuffled_labels), self.batch_size)]
        # initialize empty containers for storing the data from respective batch in the iteration
        dataX = [None]
        labelX = [None]
        # intialize scores_list for storing test loss and accuracies
        scores_list = []
        # intialize con_mat_list for storing confusion matrices
        con_mat_list = []
        # for x in range(len(data_batch)):
        # for batches of 15 videos each, test the model
        total_preds = []
        total_labels = []
        count = 1
        for x in range(len(data_batch)):
            # take xth batch test data
            dataX = data_batch[x]
            # take xth batch test labels
            labelX = label_batch[x]
            # convert labels to one-hot encoding
            labelX = to_categorical(labelX)
            # get test loss and accuracies for 32 batches of 15 videos each - total 466 videos.
            try:
                scores = model.evaluate(dataX, labelX)
                scores_list.append(scores)
            except:
                pass
            # add scores to scores_list
            # scores_list.append(scores)

            # get predictions for given data batch
            y_pred = model.predict(dataX)
            print(y_pred, labelX)
            # convert to vector
            y_pred = numpy.argmax(y_pred,axis=1)
            # append to total prediction list
            total_preds.append(to_categorical(y_pred))
            # convert to vector
            labelX = numpy.argmax(labelX, axis=1)
            # append to total label list
            total_labels.append(to_categorical(labelX))
            # create a confusion matrix for given data batch
            con_mat = tf.math.confusion_matrix(labels=labelX, predictions=y_pred).numpy()
            # store the confusion matrix in confusion_matrix_list
            con_mat_list.append(con_mat)
            # plot straight line for viewing against ROC curve
            plt.plot([0, 1], [0, 1], 'k:', alpha=0.5)
            # label
            labs = ['Test']
            # color for ROC curve
            col = 'maroon'
            # create ROC curve for current batch
            fpr, tpr, thresh = sklearn.metrics.roc_curve(labelX, y_pred)
            # plot roc curve
            # plt.scatter(thresh,'b')
            plt.plot(fpr, tpr, '-', color=col, alpha=0.7, lw=1.5, label=labs)
            # y label of ROC curve
            plt.ylabel('TPR')
            # x label of ROC curve
            plt.xlabel('FPR')
            # get area under curve
            auc_value = sklearn.metrics.auc(fpr, tpr)
            # title of roc curve --> Area under curve
            plt.title('AUC: %s '%(auc_value))
            # save roc curve
            roc_file_name = 'roc/roc' + str(count) + '.jpg'
            plt.savefig(roc_file_name)
            plt.close()
            count += 1
            # print AUC
            print("Area under curve for batch number: ",x)
            print(sklearn.metrics.auc(fpr, tpr))
            # pring accuracy for current batch
            print("Accuracy score for batch number: ",x)
            print(sklearn.metrics.accuracy_score(labelX, y_pred))  ### print accuracy for each set
            # print confusion matrix for current batch
            print("Confusion matrix for batch number: ",x)
            print(sklearn.metrics.confusion_matrix(labelX, y_pred))
            # print('Test loss:', scores[0])
        # print('Test accuracy:', scores[1])
        # total scores list
        print(scores_list)

            # for confusion matrix in con_mat_list
        count = 1
        for con in con_mat_list:
                # round the values of confusion matrix
            # con = numpy.around(con.astype('float') / con_mat.sum(axis=1)[:, numpy.newaxis], decimals=2)
            con = numpy.around(con.astype('float'))
                # store the confusion matrix to a pandas DataFrame
            con = pd.DataFrame(con,
                                   index=self.classes,
                                   columns=self.classes)
            print(con)
            figure = plt.figure(figsize=(8, 8))
            # plot heatmap of confusion matrix
            sns.heatmap(con, annot=True, cmap=plt.cm.Blues)
            plt.tight_layout()
            # y label of heat map
            plt.ylabel('True label')
            # x label of heat map
            plt.xlabel('Predicted label')
            # plt.show()
            # save the heat map
            count += 1
            cf_file_name = 'cf/cf'+str(count) + '.jpg'
            plt.savefig(cf_file_name)



if __name__ == "__main__":

    # initialize an AccidentPrediction object
    AP = AccidentPrediction()

    # get train_data_files names and its labels
    # train_data_files, train_labels = AP.combine_train_data('dataset/features/training')
    train_data_files, train_labels = AP.combine_train_data('data/train')
    # get train data
    train_data = AP.create_data_for_training(train_data_files)

    # get test_data_files names and its labels
    # test_data_files, test_labels = AP.combine_train_data('dataset/features/testing')
    test_data_files, test_labels = AP.combine_train_data('data/test')
    # get test dataLike
    test_data = AP.create_data_for_training(test_data_files)

    # initialize the model
    model = AP.network()
    # train the model
    trained_model, data, labels = AP.train_model(model, train_data, train_labels)
    # test the model
    AP.test_model(trained_model, test_data, test_labels)
