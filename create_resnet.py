from __future__ import print_function
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2

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

def resnet(lmdb, batch_size=256, stages=[2, 2, 2, 2], first_output=64, deploy=False):
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
    dim3=32
    dim4=32
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

  #n.loss layer is only in training and testing nets, but not in deploy net.
  if deploy==False:
    loss = L.SoftmaxWithLoss(fc, label)
    return to_proto(loss)
  #for generating the deploy net
 # else:
    #generate the input information header string
#    deploy_str='input: {}\ninput_dim: {}\ninput_dim: {}\ninput_dim: {}\ninput_dim: {}'.format('"'+input+'"', dim1, dim2, dim3, dim4)
    #assemble the input header with the net layers string.  remove the first placeholder layer from the net string.
#    return deploy_str+'\n'+'layer {'+'layer {'.join(to_proto(loss).split('layer {')[2:])

def make_net():
  with open('resnet_train.prototxt', 'w') as f:
#    print(writing train)
    print(resnet('/srv/datasets/ILSVRC2012/train_lmdb'), file=f)
  with open('resnet_test.prototxt', 'w') as f:
#    print(writing test)
    print(resnet('/srv/dataset/ILSVRC2012/test_lmdb'), file=f)
#  with open('resnet_deploy.prototxt', 'w') as f:
#    print(writing deploy)
#    print(resnet('srv/dataset/train_lmdb', deploy=True), file=f)

if __name__ == '__main__':
  make_net()
  caffe.Net('resnet_train.prototxt', caffe.TRAIN)
