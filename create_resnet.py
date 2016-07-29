from __future__ import print_function

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from pylab import *
import sys

import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

train_net_path = 'resnet_train.prototxt'
test_net_path = 'resnet_test.prototxt'
solver_config_path = 'resnet_solver.prototxt'

def conv_factory(bottom, ks, nout, stride=1, pad=0):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, bias_term=False, weight_filler=dict(type='msra'))
    batch_norm = L.BatchNorm(conv, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    return relu

def conv_factory_inverse(bottom, ks, nout, stride=1, pad=0):
    batch_norm = L.BatchNorm(bottom, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    conv = L.Convolution(relu, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, weight_filler=dict(type='msra'))
    return conv

def conv_factory_inverse_no_inplace(bottom, ks, nout, stride=1, pad=0):
    batch_norm = L.BatchNorm(bottom, in_place=False, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    relu = L.ReLU(scale, in_place=True)
    conv = L.Convolution(relu, kernel_size=ks, stride=stride,
                                num_output=nout, pad=pad, weight_filler=dict(type='msra'))
    return conv

def residual_factory1(bottom, num_filter):
    conv1 = conv_factory_inverse_no_inplace(bottom, 3, num_filter, 1, 1);
    conv2 = conv_factory_inverse(conv1, 3, num_filter, 1, 1);
    addition = L.Eltwise(bottom, conv2, operation=P.Eltwise.SUM)
    return addition

def residual_factory_proj(bottom, num_filter, stride=2):
    batch_norm = L.BatchNorm(bottom, in_place=True, param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    scale = L.Scale(batch_norm, bias_term=True, in_place=True)
    conv1 = conv_factory(scale, 3, num_filter, stride, 1);
    conv2 = L.Convolution(conv1, kernel_size=3, stride=1,
                                num_output=num_filter, pad=1, weight_filler=dict(type='msra'));
    proj = L.Convolution(scale, kernel_size=1, stride=stride,
                                num_output=num_filter, pad=0, weight_filler=dict(type='msra'));
    addition = L.Eltwise(conv2, proj, operation=P.Eltwise.SUM)
    return addition

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def resnet(lmdb, batch_size=64, stages=[2, 2, 2, 2], first_output=64, deploy=False):
#it is tricky to produce the deploy prototxt file, as the data input is not from a layer, so we have to creat a workaround
#producing training and testing prototxt files is pretty straight forward
  #n = caffe.NetSpec()
  if deploy==False:
    data, label = L.Data(source=lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=2,
            transform_param=dict(crop_size=224, mean_value=[104, 117, 123], mirror=True))
  # produce data definition for deploy net
  else:
    input="data"
    dim1=1
    dim2=1
    dim3=224
    dim4=224
    #make an empty "data" layer so the next layer accepting input will be able to take the correct blob name "data",
    #we will later have to remove this layer from the serialization string, since this is just a placeholder
    data=L.Layer()

  relu1 = conv_factory(data, 7, first_output, stride=2, pad=3)
  residual = max_pool(relu1, 3, stride=2)

  k=0
  for i in stages:
#    first_output *= 2
    for j in range(i):
      if j==0:
        if k==0:
          residual = residual_factory_proj(residual, first_output, 1)
          k+=1
        else:
          residual = residual_factory_proj(residual, first_output, 2)
      else:
        residual = residual_factory1(residual, first_output)
    first_output *= 2

  glb_pool = L.Pooling(residual, pool=P.Pooling.AVE, global_pooling=True);
  fc = L.InnerProduct(glb_pool, num_output=1000, weight_filler=dict(type='xavier'))
  acc = L.Accuracy(fc, label, include=dict(phase=getattr(caffe_pb2, 'TEST')))

  #n.loss layer is only in training and testing nets, but not in deploy net.
  if deploy==False:
    loss = L.SoftmaxWithLoss(fc, label)
    return to_proto(loss, acc)
  #for generating the deploy net
 # else:
    #generate the input information header string
#    deploy_str='input: {}\ninput_dim: {}\ninput_dim: {}\ninput_dim: {}\ninput_dim: {}'.format('"'+input+'"', dim1, dim2, dim3, dim4)
    #assemble the input header with the net layers string.  remove the first placeholder layer from the net string.
#    return deploy_str+'\n'+'layer {'+'layer {'.join(to_proto(loss).split('layer {')[2:])

def make_net():
  with open(train_net_path, 'w') as f:
#    print(writing train)
    print(resnet('/srv/datasets/ILSVRC2012/train_lmdb'), file=f)
  with open(test_net_path, 'w') as f:
#    print(writing test)
    print(resnet('/srv/datasets/ILSVRC2012/val_lmdb', batch_size=50), file=f)
#  with open('resnet_deploy.prototxt', 'w') as f:
#    print(writing deploy)
#    print(resnet('srv/dataset/train_lmdb', deploy=True), file=f)

if __name__ == '__main__':
  make_net()

  s = caffe_pb2.SolverParameter()
  s.train_net = train_net_path
  s.test_net.append(test_net_path)
  s.test_interval = 1000
  s.test_iter.append(1000)
  #s.test_iter.append(1000)
  s.max_iter = 600000
  s.type = 'SGD'
  s.base_lr = 0.01
  s.momentum = 0.9
  s.weight_decay = 5e-4
  #s.lr_policy = 'inv'
  #s.gamma = 0.0001
  #s.power = 0.75
  s.lr_policy = 'step'
  s.gamma = 0.1
  s.stepsize = 15000
  s.display = 1000
  s.snapshot = 5000
  s.snapshot_prefix = 'snapshots/resnet'

  s.solver_mode = caffe_pb2.SolverParameter.GPU
  with open(solver_config_path, 'w') as f:
      f.write(str(s))

  caffe.set_mode_gpu()
  caffe.set_device(4)
  solver = None
  solver = caffe.get_solver(solver_config_path)
  niter = 600000  # EDIT HERE increase to train for longer
  test_interval = 1000 #1niter / 100
  # losses will also be stored in the log
  train_loss = zeros(niter)
  test_acc = zeros(int(np.ceil(niter / test_interval)))

  for it in range(niter):
    solver.step(1)  # SGD by Caffe

    # store the train loss
    train_loss[it] = solver.net.blobs['SoftmaxWithLoss1'].data
    # store the output on the first test batch
    # (start the forward pass at conv1 to avoid loading new data)
    solver.test_nets[0].forward(start='Convolution1')

    if it % 100 == 0:
        print('iter :',it, ', loss :' ,train_loss[it])
    # run a full test every so often
    # (Caffe can also do this for us and write to a log, but we show here
    #  how to do it directly in Python, where more complicated things are easier.)
    if it % test_interval == 0:
        print('Iteration', it, 'testing...')
        correct = 0
        for test_it in range(1000):
            solver.test_nets[0].forward()
            correct += sum(solver.test_nets[0].blobs['InnerProduct1'].data.argmax(1)
                           == solver.test_nets[0].blobs['Data2'].data)
        test_acc[it // test_interval] = correct / 1e4
        print('iter :',it, ', acc :' ,test_acc[it // test_interval])

  _, ax1 = subplots()
  ax2 = ax1.twinx()
  ax1.plot(arange(niter), train_loss)
  ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
  ax1.set_xlabel('iteration')
  ax1.set_ylabel('train loss')
  ax2.set_ylabel('test accuracy')
  ax2.set_title('Custom Test Accuracy: {:.2f}'.format(test_acc[-1]))

  savefig('result.png')
