import torch
from torch import nn
from torch.nn import Module
import clip
from models.stylegan2.model import EqualLinear, PixelNorm, Upsample, Downsample, Attention, DeformConv2d
from torch.nn import Linear, LayerNorm, LeakyReLU, Sequential, PReLU, ELU
from PIL import Image
import torchvision.transforms as transforms

class ModulationModule(Module):
    def __init__(self, layernum):
        super(ModulationModule, self).__init__()
        self.layernum = layernum
        self.fc = Linear(512, 512)
        self.norm = LayerNorm([self.layernum, 512], elementwise_affine=False)
        # self.gamma_function = Sequential(Linear(512, 512), LayerNorm([512]), LeakyReLU(), Linear(512, 512))
        # self.beta_function = Sequential(Linear(512, 512), LayerNorm([512]), LeakyReLU(), Linear(512, 512))
        self.gamma_function = Sequential(Linear(512, 512), LayerNorm([512]), ELU(), Linear(512, 512))
        self.beta_function = Sequential(Linear(512, 512), LayerNorm([512]), ELU(), Linear(512, 512))
        # self.leakyrelu = LeakyReLU()

        self.elu = ELU()
        # self.prelu = PReLU()
        # self.multi_heads_attention = Attention()


    def forward(self, x, embedding, cut_flag):

        #print(x)
        x = self.fc(x)
        x = self.norm(x) 	
        if cut_flag == 1:
            return x
        # x = self.multi_heads_attention(x)
        gamma = self.gamma_function(embedding.float())
        beta = self.beta_function(embedding.float())
        out = x * (1 + gamma) + beta
        # out = self.leakyrelu(out)

        out = self.elu(out)
        return out

class SubHairMapper(Module):
    def __init__(self, opts, layernum):
        super(SubHairMapper, self).__init__()
        self.opts = opts
        self.layernum = layernum
        self.pixelnorm = PixelNorm()
        self.modulation_module_list = nn.ModuleList([ModulationModule(self.layernum) for i in range(5)])

        blur_kernel = [1, 3, 3, 1]
        self.upsamples = Upsample(blur_kernel)
        self.downsamples = Downsample(blur_kernel)

    def forward(self, x, embedding, cut_flag=0):
        x = self.pixelnorm(x)
        # print('x is', x.shape)
        # print('x.shape[0] is', x.shape[0])
        # print('x.shape[1] is', x.shape[1])
        # print('x.shape[2] is', x.shape[2])
        x = x.view(x.shape[0], x.shape[0], x.shape[1], x.shape[2])
        x = self.upsamples(x)
        x = self.downsamples(x)
        x = x.view(x.shape[0], x.shape[2], x.shape[3])


        for modulation_module in self.modulation_module_list:
        	x = modulation_module(x, embedding, cut_flag)        
        return x

#########################4fc##############################################################################
class Mapper(Module):

    def __init__(self, in_channel=512, out_channel=512, norm=True, num_layers=4):
        super(Mapper, self).__init__()

        layers = [PixelNorm()] if norm else []

        layers.append(EqualLinear(in_channel, out_channel, lr_mul=0.01, activation='fused_lrelu'))
        for _ in range(num_layers - 1):
            layers.append(EqualLinear(out_channel, out_channel, lr_mul=0.01, activation='fused_lrelu'))
        self.mapping = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mapping(x)
        return x
#########################4fc#########################################################################

class HairMapper(Module): 
    def __init__(self, opts):
        super(HairMapper, self).__init__()
        self.opts = opts
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.transform = transforms.Compose([transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        self.face_pool = torch.nn.AdaptiveAvgPool2d((224, 224))
        self.hairstyle_cut_flag = 0
        self.color_cut_flag = 0

        self.fc4 = Mapper(512, 512)


        if not opts.no_coarse_mapper: 
            self.course_mapping = SubHairMapper(opts, 4)
        if not opts.no_medium_mapper:
            self.medium_mapping = SubHairMapper(opts, 4)
        if not opts.no_fine_mapper:
            self.fine_mapping = SubHairMapper(opts, 4)
        if not opts.no_morefine_mapper:
            self.morefine_mapping = SubHairMapper(opts, 6)

    def gen_image_embedding(self, img_tensor, clip_model, preprocess):
        masked_generated = self.face_pool(img_tensor)
        masked_generated_renormed = self.transform(masked_generated * 0.5 + 0.5)
        return clip_model.encode_image(masked_generated_renormed)

    def forward(self, x, hairstyle_text_inputs, color_text_inputs, hairstyle_tensor, color_tensor):
        if hairstyle_text_inputs.shape[1] != 1:     # 如果发型文本描述不是单个样本
            hairstyle_embedding = self.clip_model.encode_text(hairstyle_text_inputs).unsqueeze(1).repeat(1, 18, 1).detach()
        elif hairstyle_tensor.shape[1] != 1:        # 如果发型图像数据不是单个样本
            hairstyle_embedding = self.gen_image_embedding(hairstyle_tensor, self.clip_model, self.preprocess).unsqueeze(1).repeat(1, 18, 1).detach()
        else:           # 发型文本描述和发型图像数据都是单个样本时
            hairstyle_embedding = torch.ones(x.shape[0], 18, 512).cuda()
        if color_text_inputs.shape[1] != 1:
            color_embedding = self.clip_model.encode_text(color_text_inputs).unsqueeze(1).repeat(1, 18, 1).detach()
        elif color_tensor.shape[1] != 1:
            color_embedding = self.gen_image_embedding(color_tensor, self.clip_model, self.preprocess).unsqueeze(1).repeat(1, 18, 1).detach()
        else:
            color_embedding = torch.ones(x.shape[0], 18, 512).cuda()


        if (hairstyle_text_inputs.shape[1] == 1) and (hairstyle_tensor.shape[1] == 1):  # 如果发型文本描述和发型图像数据是单个样本，不编辑发型
        	self.hairstyle_cut_flag = 1
        else:            # 文本发型描述和发型图像数据不是单个样本，hairstyle_cut_flag = 0表示执行操纵模块
        	self.hairstyle_cut_flag = 0
        if (color_text_inputs.shape[1] == 1) and (color_tensor.shape[1] == 1):
            self.color_cut_flag = 1
        else:
            self.color_cut_flag = 0

        #######################fc4##################################################################################
        for _ in range(4):
            x = self.fc4(x)
        ##########################fc4#################################################################################

        x_coarse = x[:, :4, :]
        x_medium = x[:, 4:8, :]
        x_fine = x[:, 8:12, :]
        x_morefine = x[:, 12:, :]
        # print("x=", x.shape)
        # print("x=", x_coarse.shape)
        # print("x=", x_medium.shape)
        # print("x=", x_fine.shape)
        # print("x=", x_morefine.shape)

        #######################fc4##################################################################################
        # for _ in range(4):
        #     x = self.fc4(x)
        #
        #     x_coarse = self.fc4(x_coarse)
        #     x_medium = self.fc4(x_medium)
        #     x_fine = self.fc4(x_fine)
        #     x_morefine = self.fc4(x_morefine)
        ##########################fc4#################################################################################


        if not self.opts.no_coarse_mapper:
            x_coarse = self.course_mapping(x_coarse, hairstyle_embedding[:, :4, :], cut_flag=self.hairstyle_cut_flag)
        else:
            x_coarse = torch.zeros_like(x_coarse)
        if not self.opts.no_medium_mapper:
            x_medium = self.medium_mapping(x_medium, hairstyle_embedding[:, 4:8, :], cut_flag=self.hairstyle_cut_flag)
        else:
            x_medium = torch.zeros_like(x_medium)
        if not self.opts.no_fine_mapper:
            x_fine = self.fine_mapping(x_fine, color_embedding[:, 8:12, :], cut_flag=self.color_cut_flag)
        else:
            x_fine = torch.zeros_like(x_fine)
        if not self.opts.no_morefine_mapper:
            x_morefine = self.morefine_mapping(x_morefine, color_embedding[:, 12:, :], cut_flag=self.color_cut_flag)
        else:
            x_morefine = torch.zeros_like(x_morefine)

        out = torch.cat([x_coarse, x_medium, x_fine, x_morefine], dim=1)
        return out