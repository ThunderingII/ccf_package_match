
# multi_dnn model
## 数据处理
* python prepare_data.py
* 模型输入的数据格式：feat1\t feat2\t....featn\t label\n
## train
* python main.py --mode=train

## test
* python main.py --mode=test --demo_model=1537768833

其中1537768833在origin_data_save/是已经训练好的模型参数；

origin_data_save/1537768833/checkpoints/ 存放的是模型参数；

origin_data_save/1537768833/results/ 存放的是模型训练时的log信息,以及预测结果;

origin_data_save/1537768833/summaries/ 可使用tensorboard查看，如：在data_save/1537768833/目录下，运行：tensorboard --logdir=./summaries,即可在浏览器查看loss曲线图和相关信息；
