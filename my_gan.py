import jittor as jt
from jittor import init
import argparse
import os
import numpy as np
import math
from jittor import nn

if jt.has_cuda:
    jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=1000, help='interval between image sampling')
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)
#单通道的32*32的灰度图

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)
        # nn.Linear(in_dim, out_dim)表示全连接层
        # in_dim：输入向量维度
        # out_dim：输出向量维度
        def block(in_feat, out_feat, normalize=True):  #用于定义一个层
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))# 0.8是momentum参数，控制均值和方差的移动平均值的权重
            layers.append(nn.LeakyReLU(0.2)) #激活函数是ReLu的变种，当输入小于0时，Leaky ReLU会乘以0.2，而不是直接输出0
            return layers
        self.model = nn.Sequential(*block((opt.latent_dim + opt.n_classes), 128, normalize=False), #输入维度为噪声向量维度+类别数
                                   *block(128, 256), 
                                   *block(256, 512),
                                   #最终得到一个1024维的向量 
                                   *block(512, 1024), 
                                   #重新构建成32*32的图像矩阵
                                   nn.Linear(1024, int(np.prod(img_shape))), 
                                   #激活函数tanh将输出限制在-1到1之间，即起到激活作用（因为后面还要接判别器的第一层），又起到归一作用
                                   nn.Tanh())

    def execute(self, noise, labels):
        gen_input = jt.contrib.concat((self.label_emb(labels), noise), dim=1)#将噪声和label在第一维度上拼接
        img = self.model(gen_input)
        # 将img从1024维向量变为32*32矩阵
        img = img.view((img.shape[0], *img_shape))
        return img

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)
        self.model = nn.Sequential(nn.Linear((opt.n_classes + int(np.prod(img_shape))), 512), #输入维度为图像向量维度+类别数（即label）
                                   nn.LeakyReLU(0.2), #激活函数，当输入小于0时，Leaky ReLU会乘以0.2，而不是直接输出0
                                   nn.Linear(512, 512), 
                                   nn.Dropout(0.4), 
                                   nn.LeakyReLU(0.2), 
                                   nn.Linear(512, 512), 
                                   nn.Dropout(0.4), #dropout层，防止过拟合，丢弃概率为0.4
                                   nn.LeakyReLU(0.2), 
                                   # TODO: 添加最后一个线性层，最终输出为一个实数
                                   nn.Linear(512, 1),
                                   nn.Sigmoid() #归一
                                   )

    def execute(self, img, labels):
        #前1024列是图像，后10列是标签
        d_in = jt.contrib.concat((img.view((img.shape[0], (- 1))), self.label_embedding(labels)), dim=1)
        # TODO: 将d_in输入到模型中并返回计算结果
        validity = self.model(d_in)
        return validity #返回判别器的输出,即真实图片的概率(归一化后在0-1之间)

# 损失函数：平方误差
# 调用方法：adversarial_loss(网络输出A, 分类标签B)
# 计算结果：(A-B)^2
adversarial_loss = nn.MSELoss()

generator = Generator()
discriminator = Discriminator()

# 导入MNIST数据集
from jittor.dataset.mnist import MNIST
import jittor.transform as transform
transform = transform.Compose([
    transform.Resize(opt.img_size), #将图片缩放到指定大小
    transform.Gray(), #将图片转换为灰度图，单通道
    transform.ImageNormalize(mean=[0.5], std=[0.5]),#进行标准化处理，将像素值缩放到指定的均值和标准差范围内，使其接近标准正态分布
])
dataloader = MNIST(train=True, transform=transform).set_attrs(batch_size=opt.batch_size, shuffle=True)

optimizer_G = nn.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = nn.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

G_loss = []
D_loss = []

from PIL import Image
def save_image(img, path, nrow=10, padding=5):
    N,C,W,H = img.shape
    if (N%nrow!=0):
        print("N%nrow!=0")
        return
    ncol=int(N/nrow)
    img_all = []
    for i in range(ncol):
        img_ = []
        for j in range(nrow):
            img_.append(img[i*nrow+j])
            img_.append(np.zeros((C,W,padding)))
        img_all.append(np.concatenate(img_, 2))
        img_all.append(np.zeros((C,padding,img_all[0].shape[2])))
    img = np.concatenate(img_all, 1)
    img = np.concatenate([np.zeros((C,padding,img.shape[2])), img], 1)
    img = np.concatenate([np.zeros((C,img.shape[1],padding)), img], 2)
    min_=img.min()
    max_=img.max()
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img = img[:,:,::-1]
    elif C==1:
        img = img[:,:,0]
    Image.fromarray(np.uint8(img)).save(path)

def sample_image(n_row, batches_done):
    # 随机采样输入并保存生成的图片，latent_dim指定了噪声向量的维度
    z = jt.array(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))).float32().stop_grad()
    #labels是要保存的数字序列，十行0，1，...,9
    labels = jt.array(np.array([num for _ in range(n_row) for num in range(n_row)])).float32().stop_grad()
    gen_imgs = generator(z, labels) #根据序列生成图像，z是噪声，labels是数字序列
    save_image(gen_imgs.numpy(), "%d.png" % batches_done, nrow=n_row)

# ----------
#  模型训练
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0] #64

        # 数据标签，valid=1表示真实的图片，fake=0表示生成的图片
        valid = jt.ones([batch_size, 1]).float32().stop_grad()
        fake = jt.zeros([batch_size, 1]).float32().stop_grad()

        # 真实图片及其类别
        real_imgs = jt.array(imgs)
        labels = jt.array(labels)

        # -----------------
        #  训练生成器
        # -----------------

        # 采样随机噪声和数字类别作为生成器输入
        z = jt.array(np.random.normal(0, 1, (batch_size, opt.latent_dim))).float32() #随机的噪声
        gen_labels = jt.array(np.random.randint(0, opt.n_classes, batch_size)).float32()  #随机的数字类别（label）

        # 生成一组图片
        gen_imgs = generator(z, gen_labels) 
        # 损失函数衡量生成器欺骗判别器的能力，即希望判别器将生成图片分类为valid
        validity = discriminator(gen_imgs, gen_labels)  #判别器判断生成的图片的真实性
        g_loss = adversarial_loss(validity, valid) #计算生成器的损失，即生成的图片被判别器判断为真实图片的概率与1的差值
        g_loss.sync() #同步生成器损失（g_loss）的值，以确保所有进程都使用相同的损失值进行更新，从而保持模型的一致性
        optimizer_G.step(g_loss) #更新生成器的参数
        #生成器的目的是生成尽可能逼真的图片，使得判别器无法区分真实图片和生成图片，让validity尽可能接近1

        # ---------------------
        #  训练判别器
        # ---------------------

        validity_real = discriminator(real_imgs, labels)  #判别器判断真实图片的真实性
        d_real_loss = adversarial_loss(validity_real, valid) #计算真实图片的损失，目的要使真实图片分类为valid，real要不断接近1

        validity_fake = discriminator(gen_imgs.stop_grad(), gen_labels) #判别器判断生成的图片的真实性
        d_fake_loss = adversarial_loss(validity_fake, fake) #目的要让生成的图片分类为fake，fake要不断接近0

        # 总的判别器损失
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.sync() #同步判别器损失（d_loss）的值
        optimizer_D.step(d_loss) #更新判别器的参数
        

        if i  % 50 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.data, g_loss.data)
            )
            G_loss.append(g_loss.data[0])
            D_loss.append(d_loss.data[0])

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)

    if epoch % 10 == 0:
        generator.save("generator_last.pkl")
        discriminator.save("discriminator_last.pkl")

generator.eval()
discriminator.eval()
generator.load('generator_last.pkl')
discriminator.load('discriminator_last.pkl')

number = str(20603842055512)#TODO: 写入比赛页面中指定的数字序列（字符串类型）
n_row = len(number)
z = jt.array(np.random.normal(0, 1, (n_row, opt.latent_dim))).float32().stop_grad()
labels = jt.array(np.array([int(number[num]) for num in range(n_row)])).float32().stop_grad()
gen_imgs = generator(z,labels)

img_array = gen_imgs.data.transpose((1,2,0,3))[0].reshape((gen_imgs.shape[2], -1))
min_=img_array.min()
max_=img_array.max()
img_array=(img_array-min_)/(max_-min_)*255
Image.fromarray(np.uint8(img_array)).save("result.png")

import seaborn as sns
import matplotlib.pyplot as plt

sns.lineplot(G_loss, label='Generator Loss')
sns.lineplot(D_loss, label='Discriminator Loss')

plt.show()