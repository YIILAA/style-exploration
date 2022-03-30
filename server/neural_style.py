# 风格转换的API
# mostly borrowed from https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/vgg.py

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils
from transformer_net import TransformerNet
from vgg import Vgg16


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
                for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


# web应用接口
# 暂时只有一种id
def get_stylize(content_path, style_ids, style_num, model_path): # style_num如何确定？
    print("get into funciton get_stylize")
    # output_path = './images/output_web_app/'+sessionId+'.png'
    imname = content_path.split('/')[-1] # sessionId+'.png'
    # device
    # todo：有gpu时用第一个，后续可改成多gpu
    device = try_gpu()

    # content_image
    content_image = utils.load_image(content_path)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device) # 增加一个维度

    # style_ids 
    style_ids = list(map(int, style_ids.split(',')))
    style_ids = torch.LongTensor(style_ids).to(device)

    '''
    style_blend_weights = [1.0]
    style_blend_weights = torch.tensor(style_blend_weights).to(device)
    '''
    # 处理style_blend_weights列表
    # todo 暂时是平均的
    style_blend_weights = []
    for i in style_ids:
            style_blend_weights.append(1.0)
    # Normalize the style blending weights so they sum to 1
    style_blend_weights = torch.Tensor(style_blend_weights)
    style_blend_weights /= torch.sum(style_blend_weights)
    style_blend_weights = style_blend_weights.to(device)

    # 处理
    with torch.no_grad():
        import time
        start = time.time()
        # 加载模型
        style_model = TransformerNet(style_num=style_num) # 模型结构
        state_dict = torch.load(model_path) # 预训练的参数
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        # forward
        output = style_model(content_image, style_ids, blend=True, style_blend_weights=style_blend_weights).cpu()
        # 计时
        end = time.time()
        print('Time={}'.format(end - start))
    
    # 如何处理输出图片
    # test - save
    utils.save_image(os.path.join('./images/output_web_app', imname), output[0])

def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize(args.image_size), # the shorter side is resize to match image_size
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(), # to tensor [0,1]
        transforms.Lambda(lambda x: x.mul(255)) # convert back to [0, 255]
    ])
    train_dataset = datasets.ImageFolder(args.dataset, transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True) # to provide a batch loader

    # 需要用style_num初始化网络
    # style_image = [f for f in os.listdir(args.style_image)] # 读取顺序是怎么样的？？？ -- os.listdir() 顺序不一定
    # 要确定顺序的话需要加一个排序。文件名待确认
    style_image = os.listdir(args.style_image) 
    style_image.sort() # 按照文件名排序
    style_num = len(style_image)
    print(args.style_image)
    print(style_image)
    print(style_num)

    transformer = TransformerNet(style_num=style_num).to(device) 
    optimizer = Adam(transformer.parameters(), args.lr)
    mse_loss = torch.nn.MSELoss()

    # ============================= 获取需要匹配的style gram特征 =============================
    vgg = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.Resize(args.style_size), 
        transforms.CenterCrop(args.style_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)) # 为什么先*255后由除掉呢？
    ])

    style_batch = []

    for i in range(style_num):
        style = utils.load_image(os.path.join(args.style_image, style_image[i]), size=args.style_size)
        style = style_transform(style)
        style_batch.append(style) # 列表，元素为一张图的tensor

    style = torch.stack(style_batch).to(device) # size: (n * c * h * w)

    features_style = vgg(utils.normalize_batch(style)) # 类型 namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
    gram_style = [utils.gram_matrix(y) for y in features_style] # y.shape: (n * c*h*w)
    # gram_style 列表大小为(4,), 元素为gram矩阵(n*c*c)

    # ============================= 训练 =============================
    for e in range(args.epochs):
        transformer.train()
        agg_content_loss = 0.
        agg_style_loss = 0.
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x) # x的第一维，当前batch中数据量
            
            if n_batch < args.batch_size:
                break # skip to next epoch when no enough images left in the last batch of current epoch

            count += n_batch
            optimizer.zero_grad() # initialize with zero gradients

            batch_style_id = [i % style_num for i in range(count-n_batch, count)] # 这个batch训练时每张图对应的style依次循环
            # 为一个列表，指定batch中每张图对应的style_id
            batch_style_id = torch.LongTensor(batch_style_id).to(device)
            y = transformer(x.to(device), style_id = batch_style_id) # 网络的一次前向需要X，X对应的style_id列表
            # x为(batch*c*h*w), style_id为(batch,)

            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)

            features_y = vgg(y.to(device))
            features_x = vgg(x.to(device))
            content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)

            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style): # 生成图像的一层的features(batch*c*h*w) 所有风格其中一层的gram矩阵(n*c*c)
                gm_y = utils.gram_matrix(ft_y) # 计算生成图像当前层的gram矩阵(batch*c*c)
                style_loss += mse_loss(gm_y, gm_s[batch_style_id, :, :]) # batch中每张图gram矩阵 和 对应风格的gram矩阵计算mse_loss
            style_loss *= args.style_weight

            # 不计算smooth损失吗？
            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()

            if (batch_id + 1) % args.log_interval == 0:
                mesg = "{}\tEpoch {}:\t[{}/{}]\tcontent: {:.6f}\tstyle: {:.6f}\ttotal: {:.6f}".format(
                    time.ctime(), e + 1, count, len(train_dataset),
                                  agg_content_loss / (batch_id + 1),
                                  agg_style_loss / (batch_id + 1),
                                  (agg_content_loss + agg_style_loss) / (batch_id + 1)
                )
                print(mesg)

            if args.checkpoint_model_dir is not None and (batch_id + 1) % args.checkpoint_interval == 0:
                transformer.eval().cpu()
                ckpt_model_filename = "ckpt_epoch_" + str(e) + "_batch_id_" + str(batch_id + 1) + ".pth"
                ckpt_model_path = os.path.join(args.checkpoint_model_dir, ckpt_model_filename)
                torch.save(transformer.state_dict(), ckpt_model_path)
                transformer.to(device).train()

    # ============================= save model =============================
    transformer.eval().cpu()
    save_model_filename = "epoch_" + str(args.epochs) + "_" + str(time.ctime()).replace(' ', '_').replace(':', '') + "_" + str(int(
        args.content_weight)) + "_" + str(int(args.style_weight)) + ".model"
    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
    torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)


# main文件调用
def stylize(args):
    # 修改成传入列表形式的，style_id，style_blend_weights，需要一个combine的bool型参数
    device = torch.device("cuda" if args.cuda else "cpu")

    content_image = utils.load_image(args.content_image, scale=args.content_scale)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device) # 增加一个维度
    # style_id = torch.LongTensor([args.style_id]).to(device) # 格式为列表，但仅含一个元素

    # 处理style_ids
    # 得到style_ids列表
    style_ids = list(map(int, args.style_ids.split(',')))
    style_ids = torch.LongTensor(style_ids).to(device)

    # 处理style_blend_weights列表
    style_blend_weights = []
    if args.style_blend_weights == None:
        # 未指定blend_weights，使用相同的权重
        for i in style_ids:
            style_blend_weights.append(1.0)
    else:
        # 指定，检查长度是否一致
        style_blend_weights = list(map(float, args.style_blend_weights.split(',')))
        assert len(style_blend_weights) == len(style_ids), \
            "--style-blend-weights and --style-ids must have the same number of elements!"
    # Normalize the style blending weights so they sum to 1
    style_blend_weights = torch.Tensor(style_blend_weights)
    style_blend_weights /= torch.sum(style_blend_weights)
    style_blend_weights = style_blend_weights.to(device)

    with torch.no_grad():
        import time
        start = time.time()
        # 加载模型
        style_model = TransformerNet(style_num=args.style_num)
        state_dict = torch.load(args.model)
        style_model.load_state_dict(state_dict)
        style_model.to(device)
        # forward
        output = style_model(content_image, style_ids, blend=True, style_blend_weights=style_blend_weights).cpu() # 修改模型处理 需要融合的forward
        end = time.time()
        print('Time={}'.format(end - start))
    '''
    if args.export_onnx:
        assert args.export_onnx.endswith(".onnx"), "Export model file should end with .onnx"
        output = torch.onnx._export(style_model, [content_image_t,style_t], args.export_onnx, input_names=['input_image','style_index'], output_names=['output_image']).cpu()
    '''

    # save
    output_image_dir = 'images/output_images/'
    output_image_name = args.output_image+ '_style'+str(args.style_ids)
    if args.style_blend_weights is not None:
        output_image_name += '_weight'+str(args.style_blend_weights)
    utils.save_image(output_image_dir + output_image_name +'.jpg', output[0])

def main():
    '''根据命令行输入构造args'''
    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=1,
                                  help="number of training epochs, default is 2")
    train_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for training, default is 4")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--style-image", type=str, default="images/style-images/",
                                  help="path to style-image")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=256,
                                  help="size of training images, default is 256 X 256")
    train_arg_parser.add_argument("--style-size", type=int, default=512,
                                  help="size of style-image, default is the original size of style image")
    # todo：如果有gpu就使用gpu，函数try_all_gpus()
    train_arg_parser.add_argument("--cuda", type=int, required=True,
                                  help="set it to 1 for running on GPU, 0 for CPU")
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--content-weight", type=float, default=1e5,
                                  help="weight for content-loss, default is 1e5")
    train_arg_parser.add_argument("--style-weight", type=float, default=1e10,
                                  help="weight for style-loss, default is 1e10")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")
    train_arg_parser.add_argument("--log-interval", type=int, default=250,
                                  help="number of images after which the training loss is logged, default is 250")
    train_arg_parser.add_argument("--checkpoint-interval", type=int, default=1000,
                                  help="number of batches after which a checkpoint of the trained model will be created")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--content-image", type=str, required=True,
                                 help="path to content image you want to stylize")
    eval_arg_parser.add_argument("--content-scale", type=float, default=None,
                                 help="factor for scaling down the content image")
    eval_arg_parser.add_argument("--output-image", type=str, required=True,
                                 help="path for saving the output image")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="saved model to be used for stylizing the image. If file ends in .pth - PyTorch path is used, if in .onnx - Caffe2 path")
    # todo：如果有gpu就使用gpu，函数try_all_gpus()
    eval_arg_parser.add_argument("--cuda", type=int, required=True,
                                 help="set it to 1 for running on GPU, 0 for CPU")
    '''
    eval_arg_parser.add_argument("--style-id", type=int, required = True,
                                 help="style number id corresponding to the order in training")
    '''
    # new
    eval_arg_parser.add_argument("--style-ids",
                                help="style number id corresponding to the order in training, could be more than one id split by ','")
    eval_arg_parser.add_argument("--style-blend-weights", 
                                help="")
    eval_arg_parser.add_argument("--batch-size", type=int, default=4,
                                  help="batch size for testing, default is 4")
    eval_arg_parser.add_argument("--style-num", type=int, default=4,
                                  help="number of styles used in training, default is 4")
    eval_arg_parser.add_argument("--export_onnx", type=str,
                                 help="export ONNX model to a given file")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1) # 1 for all other types of error besides syntax
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        check_paths(args)
        train(args)
    else:
        stylize(args)

if __name__ == "__main__":
    main()

    # 测试函数
    # get_stylize('images/content_images/chicago.jpg', [0], 32, 'pytorch_models/style_model.pth')