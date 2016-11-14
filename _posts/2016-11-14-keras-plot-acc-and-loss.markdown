---
layout: post
tags: [Others]
---
Keras是一个高层神经网络库，Keras由纯Python编写而成并基Tensorflow或Theano。Keras 为支持快速实验而生，能够把你的idea迅速转换为结果，如果你有如下需求，请选择Keras：

* 简易和快速的原型设计（keras具有高度模块化，极简，和可扩充特性）
* 支持CNN和RNN，或二者的结合
* 支持任意的链接方案（包括多输入和多输出训练）
* 无缝CPU和GPU切换

Keras绘制精度和损失曲线使用Keras中的回调函数Callback。具体代码如下所示：

{% highlight python %}
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type, filename=None):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        fig_name = filename + '-' + loss_type + '-acc-loss.png'
        plt.savefig(fig_name, format='png')
        plt.show()
{% endhighlight %}

{% highlight python %}
mnist_Model_DA.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
history_DA = LossHistory()
checkpointer = keras.callbacks.ModelCheckpoint(filepath="bestModel.hdf5", verbose=1, save_best_only=True) 

mnist_Model_DA.fit(X_train_DA, Y_train_DA, 
					batch_size=batch_size, 
					nb_epoch=nb_epoch,
					verbose=1, 
					validation_data=(X_test, Y_test),
					callbacks=[checkpointer, history_DA])
{% endhighlight %}
更多关于回调函数请点击[这里](http://keras-cn.readthedocs.io/en/latest/other/callbacks/)

更多Keras使用方法请查看手册
* [中文手册](http://keras-cn.readthedocs.io/en/latest/)
* [英文手册](https://keras.io/)
* [github](https://github.com/fchollet/keras)

