"""
@author: Tejas Arya (ta2763)
@author: Amritha Venkataramana (axv3602)

"""
import matplotlib.pyplot as plt
arr = [[0.847466766834259, 0.46666666865348816], [0.6447384357452393, 0.5333333611488342], [0.7249559164047241, 0.6000000238418579]]

loss = []
accuracy = []
for each in arr:
    loss.append(each[0])
    accuracy.append(each[1]*100)
y = [x for x in range(len(loss))]
plt.plot(y,loss)
plt.plot(y, loss, 'r.')
plt.xlabel('batches')
plt.ylabel('loss')
plt.show()
plt.close()
plt.xlabel('batches')
plt.ylabel('accuracy')
plt.plot(y, accuracy, 'r.')
plt.plot(y,accuracy)
plt.show()