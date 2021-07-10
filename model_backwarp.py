import torch.nn as nn
import torch
import getopt
import math
import sys
from utils import loop_pic,loop_feature

class LateralBlock(nn.Module):
	def __init__(self, ch_in, ch_out):
		super().__init__()
		self.f = nn.Sequential(
			nn.PReLU(),
			nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
			nn.PReLU(),
			nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
		)
		if ch_in != ch_out:
			self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)

	def forward(self, x):
		fx = self.f(x)
		if fx.shape[1] != x.shape[1]:
			x = self.conv(x)

		return fx + x


class DownSamplingBlock(nn.Module):

	def __init__(self, ch_in, ch_out):
		super().__init__()
		self.f = nn.Sequential(
			nn.PReLU(),
			nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
			nn.PReLU(),
			nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
		)


	def forward(self, x):
		return self.f(x)


class UpSamplingBlock(nn.Module):

	def __init__(self, ch_in, ch_out):
		super().__init__()
		self.f = nn.Sequential(
			nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
			# nn.UpsamplingNearest2d(scale_factor = 2),
			nn.PReLU(),
			nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
			nn.PReLU(),
			nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
		)

	def forward(self, x):
		return self.f(x)

try:
	from .correlation import correlation # the custom cost volume layer
except:
	sys.path.insert(0, './correlation'); import correlation # you should consider upgrading python
# end

##########################################################
torch.set_grad_enabled(True) # make sure to not compute gradients for computational performance
torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'default'
for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--model' and strArgument != '': arguments_strModel = strArgument # which model to use


class pwcnet(torch.nn.Module):
	def __init__(self):
		super(pwcnet, self).__init__()

		class Extractor(torch.nn.Module):
			def __init__(self):
				super(Extractor, self).__init__()

				self.netOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)
			# end

			def forward(self, tenInput):
				tenOne = self.netOne(tenInput)
				tenTwo = self.netTwo(tenOne)
				tenThr = self.netThr(tenTwo)
				tenFou = self.netFou(tenThr)
				tenFiv = self.netFiv(tenFou)
				tenSix = self.netSix(tenFiv)

				return [ tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix ]
			# end
		# end

		class Decoder(torch.nn.Module):
			def __init__(self, intLevel):
				super(Decoder, self).__init__()

				intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 1]
				intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 0]

				if intLevel < 6: self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.fltBackwarp = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]

				self.netOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
				)
			# end
			def backwarp(self,tenInput, tenFlow):

				tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1).type_as(tenInput)
				tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3]).type_as(tenInput)

				backwarp_tenGrid = torch.cat([ tenHor, tenVer ], 1)
				# end


				backwarp_tenPartial = tenFlow.new_ones([ tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3] ])
				# end

				tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
				tenInput = torch.cat([ tenInput, backwarp_tenPartial ], 1)


				tenOutput = torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)

				tenMask = tenOutput[:, -1:, :, :]; tenMask[tenMask > 0.999] = 1.0; tenMask[tenMask < 1.0] = 0.0

				return tenOutput[:, :-1, :, :] * tenMask

			def forward(self, tenFirst, tenSecond, objPrevious):
				tenFlow = None
				tenFeat = None

				if objPrevious is None:
					tenFlow = None
					tenFeat = None

					tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenFirst=tenFirst, tenSecond=tenSecond), negative_slope=0.1, inplace=False)

					tenFeat = torch.cat([ tenVolume ], 1)

				elif objPrevious is not None:
					tenFlow = self.netUpflow(objPrevious['tenFlow'])
					tenFeat = self.netUpfeat(objPrevious['tenFeat'])

					tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenFirst=tenFirst, tenSecond=self.backwarp(tenInput=tenSecond, tenFlow=tenFlow * self.fltBackwarp)), negative_slope=0.1, inplace=False)

					tenFeat = torch.cat([ tenVolume, tenFirst, tenFlow, tenFeat ], 1)

				# end

				tenFeat = torch.cat([ self.netOne(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netTwo(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netThr(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netFou(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netFiv(tenFeat), tenFeat ], 1)

				tenFlow = self.netSix(tenFeat)

				return {
					'tenFlow': tenFlow,
					'tenFeat': tenFeat
				}
			# end
		# end

		class Refiner(torch.nn.Module):
			def __init__(self):
				super(Refiner, self).__init__()

				self.netMain = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
				)
			# end

			def forward(self, tenInput):
				return self.netMain(tenInput)
			# end
		# end

		self.netExtractor = Extractor()

		self.netTwo = Decoder(2)
		self.netThr = Decoder(3)
		self.netFou = Decoder(4)
		self.netFiv = Decoder(5)
		self.netSix = Decoder(6)

		self.netRefiner = Refiner()

		self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-pwc/network-' + arguments_strModel + '.pytorch', file_name='pwc-' + arguments_strModel).items() })
	# end

	def forward(self, tenFirst, tenSecond):
		tenFirst = self.netExtractor(tenFirst)
		tenSecond = self.netExtractor(tenSecond)

		objEstimate = self.netSix(tenFirst[-1], tenSecond[-1], None)
		objEstimate = self.netFiv(tenFirst[-2], tenSecond[-2], objEstimate)
		objEstimate = self.netFou(tenFirst[-3], tenSecond[-3], objEstimate)
		objEstimate = self.netThr(tenFirst[-4], tenSecond[-4], objEstimate)
		objEstimate = self.netTwo(tenFirst[-5], tenSecond[-5], objEstimate)

		return objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])

class Feature_extractor(torch.nn.Module):
	def __init__(self):
		super(Feature_extractor, self).__init__()

		self.conv1 = torch.nn.Sequential(
			nn.Conv2d(
				in_channels=3,
				out_channels=64,
				kernel_size=7,
				stride=1,
				padding=3
			),
			nn.ReLU(inplace=True),
		)
		# self.instance_norm = nn.InstanceNorm2d(128)

	def forward(self, frame1, frame3,batch_size):

		feature_1 = self.conv1(frame1)
		feature_2 = self.conv1(frame3)

		norm_feature_1,norm_feature_2=loop_feature(feature_1,feature_2,batch_size)

		# do instance norm for extract features
		# concate_feature=torch.cat((feature_1, feature_2), 1)
		# instance_norm_result=self.instance_norm(concate_feature)
		# feature_1=instance_norm_result[:,0:64,:,:]
		# feature_2=instance_norm_result[:,64:128,:,:]
		print('***')
		return norm_feature_1.type_as(frame1),norm_feature_2.type_as(frame1)

class warp(torch.nn.Module):
	def __init__(self):
		super(warp, self).__init__()
		self.get_flow = pwcnet()
		self.get_feature = Feature_extractor()
	def backwarp(self, tenInput, tenFlow):
		"""tenFlow is optical-flow import from pwc-net
		   tenInput is one input original images
		   use two images to generate optical-flow and warping base on one original imput image
		"""
		tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(
			1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1).type_as(tenInput)
		tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(
			1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3]).type_as(tenInput)

		backwarp_tenGrid = torch.cat([tenHor, tenVer], 1)

		tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
							 tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)
		return torch.nn.functional.grid_sample(input=tenInput ,
											   grid=(backwarp_tenGrid+ tenFlow).permute(0, 2, 3, 1) ,
											   mode='bilinear', padding_mode='zeros', align_corners=False)

	def prepare_for_backwarp(self,input1,input2):
		batch_size = input1.shape[0]
		frame_width = input1.shape[3]
		frame_height = input1.shape[2]
		intPreprocessedWidth = int(math.floor(math.ceil(frame_width / 64.0) * 64.0))
		intPreprocessedHeight = int(math.floor(math.ceil(frame_height / 64.0) * 64.0))

		tenPreprocessedFirst = input1.view(batch_size, 3, frame_height, frame_width)
		tenPreprocessedSecond = input2.view(batch_size, 3, frame_height, frame_width)

		tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenPreprocessedFirst,
																 size=(intPreprocessedHeight, intPreprocessedWidth),
																 mode='bilinear', align_corners=False)
		tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenPreprocessedSecond,
																  size=(intPreprocessedHeight, intPreprocessedWidth),
																  mode='bilinear', align_corners=False)
		tenFlow = 20.0 * torch.nn.functional.interpolate(
			input=self.get_flow(tenPreprocessedFirst, tenPreprocessedSecond),
			size=(frame_height, frame_width), mode='bilinear',
			align_corners=False)
		tenFlow[:, 0, :, :] *= float(frame_width) / float(intPreprocessedWidth)
		tenFlow[:, 1, :, :] *= float(frame_height) / float(intPreprocessedHeight)
		flow = tenFlow[:, :, :, :]
		return flow


	def forward(self,frame1, frame3,batch_size):

		flow1=self.prepare_for_backwarp(frame1, frame3)
		flow2=self.prepare_for_backwarp(frame3, frame1)
		feature_1, feature_2 = self.get_feature(frame1, frame3,batch_size)
		feature_forward = self.backwarp(tenInput=feature_1, tenFlow=flow1 * 0.5)
		pic_forward = self.backwarp(tenInput=frame1, tenFlow=flow1 * 0.5)
		pic_backward = self.backwarp(tenInput=frame3, tenFlow=flow2 * 0.5)
		feature_backward = self.backwarp(tenInput=feature_2, tenFlow=flow2 * 0.5)
		concat=torch.cat((feature_forward,pic_forward,pic_backward,feature_backward),1)
		return concat


class GridNet(torch.nn.Module):
	def __init__(self, in_chs, out_chs, grid_chs=[32, 64, 96]):
		super().__init__()
		self.n_row = 3
		self.n_col = 6
		self.n_chs = grid_chs
		self.lateral_init = LateralBlock(in_chs, self.n_chs[0])
		self.get_input = warp()


		for r, n_ch in enumerate(self.n_chs):
			for c in range(self.n_col - 1):
				setattr(self, f'lateral_{r}_{c}', LateralBlock(n_ch, n_ch))

		for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[:-1], self.n_chs[1:])):
			for c in range(int(self.n_col / 2)):
				setattr(self, f'down_{r}_{c}', DownSamplingBlock(in_ch, out_ch))

		for r, (in_ch, out_ch) in enumerate(zip(self.n_chs[1:], self.n_chs[:-1])):
			for c in range(int(self.n_col / 2)):
				setattr(self, f'up_{r}_{c}', UpSamplingBlock(in_ch, out_ch))

		self.lateral_final = LateralBlock(self.n_chs[0], out_chs)

	def forward(self, frame1,frame3,batch_size):
		print(1,frame1.device)
		# --- instance norm --- #
		after_norm1,after_norm3,mean1,std1=loop_pic(frame1,frame3,batch_size)
		x=self.get_input(after_norm1.type_as(frame1), after_norm3.type_as(frame1),batch_size)
		state_00 = self.lateral_init(x)
		state_10 = self.down_0_0(state_00)
		state_20 = self.down_1_0(state_10)

		state_01 = self.lateral_0_0(state_00)
		state_11 = self.down_0_1(state_01) + self.lateral_1_0(state_10)
		state_21 = self.down_1_1(state_11) + self.lateral_2_0(state_20)

		state_02 = self.lateral_0_1(state_01)
		state_12 = self.down_0_2(state_02) + self.lateral_1_1(state_11)
		state_22 = self.down_1_2(state_12) + self.lateral_2_1(state_21)

		state_23 = self.lateral_2_2(state_22)
		state_13 = self.up_1_0(state_23) + self.lateral_1_2(state_12)
		state_03 = self.up_0_0(state_13) + self.lateral_0_2(state_02)

		state_24 = self.lateral_2_3(state_23)
		state_14 = self.up_1_1(state_24) + self.lateral_1_3(state_13)
		state_04 = self.up_0_1(state_14) + self.lateral_0_3(state_03)

		state_25 = self.lateral_2_4(state_24)
		state_15 = self.up_1_2(state_25) + self.lateral_1_4(state_14)
		state_05 = self.up_0_2(state_15) + self.lateral_0_4(state_04)
		output = self.lateral_final(state_05)
		# --- reverse instance norm --- #
		output = (output * std1.type_as(frame1)) + mean1.type_as(frame1)
		return output




