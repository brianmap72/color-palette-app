import os
import numpy as np
import cv2
import colorsys
from sklearn.cluster import KMeans

class TopColorKMeans():
	def __init__(self, n_clusters=5, max_iter=100, random_state=None, sort=True, resize=True):
		'''
		Parameters: 
			n_clusters:int=5, number of top colors.
			max_iter:int=100, maximum number of iteration for k means training
			sort:bool=True, whether to sort the output result in descending order
			resize:bool=True, whether to resize the image for faster performance, resizing to (256, 256)
		'''
		self.model = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=None)
		self.sort = sort
		self.resize = resize

	def get_top_color(self, img):
		''' 
		Call this function to get top colors. Percentage return in range [0,1]
		Parameters: img:ndarray 
		Return: percentage:[float], hex_color:[str], color_variation:[[str]]
		'''
		img = self._process_img(img)
		self.model.fit(img)
		pct, rgb_color, hex_color = self._get_cluster_details()
		return pct, hex_color, self._get_color_varaition(rgb_color)

	def _process_img(self, img):
		if img.ndim is not 2: 
			if self.resize:  img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
			img = img.reshape((img.shape[0]*img.shape[1], 3))
		return img

	def _get_cluster_details(self):
		unique, counts = np.unique(self.model.labels_, return_counts=True)
		pct = np.array([c/counts.sum() for c in counts])
		rgb_color = self.model.cluster_centers_.round().astype(int)
		if self.sort: pct, rgb_color = pct[pct.argsort()][::-1], rgb_color[pct.argsort()][::-1]
		hex_color = [self.rgb2hex(*c).upper() for c in rgb_color]
		return pct, rgb_color, hex_color

	def rgb2hex(self, r, g, b):
		'''
		Convert rgb to hex
		Parameters: 
			r:int, color of r channel
			g:int, color of g channel
			b:int, color of b channel
		Return: 
			hex_color:string, the converted hex
		'''
		return "#{:02x}{:02x}{:02x}".format(r, g, b)

	def hex2rgb(self, hex_color):
		'''
		Convert hex string to rgb
		Parameters: 
			hex_color:str, the hex color to be converted in the format of either '#A1B2C3' or 'D4E5F6'
		Return: 
			rgb_color:tuple, the converted rgb
		'''
		if len(hex_color) == 7 and hex_color[0] == '#': hex_color = hex_color[1:]
		return tuple([int(hex_color[i:i+2], 16) for i in range(0, 6, 2)])

	def _color_variation(self, color, var=2):
		'''
		Calculate for colour variation such as complementaary, triadic and tetradic color
		Parameters: 
			color:[int], input color with shape of (n, w, h, c)
			var:int=2, value that determines the degree of movement of H value in HSL
			{2: complementary color, 3: triadic color, 4: tetradic color}
		Return: 
			color:[int], output color with shape of (m, n, w, h, c) if var > 2
			where m is the number of possible variations
			output color with shape of (n, w, h, c) if var == 2
		'''
		norm_color = color/255
		hls_list_duplicate = [[colorsys.rgb_to_hls(*c) for c in norm_color] for i in range(var-1)]
		hls_cvt = [[[h+((1/var)*(i+1)), l, s] for h, l, s in hls_list] for i, hls_list in enumerate(hls_list_duplicate)]
		rgb_clr = np.array([[colorsys.hls_to_rgb(*c) for c in hc] for hc in hls_cvt])
		rgb_denorm = np.array([[c * 255 for c in rc] for rc in rgb_clr]).round().astype(int)
		return rgb_denorm

	def _get_color_varaition(self, rgb_color):
		color_dict = {}
		for i in range(2,5):
			res = self._color_variation(rgb_color, var=i)
			res_hex = [[self.rgb2hex(*n) for n in m] for m in res]
			color_dict['var_'+str(i)] = res_hex
		return color_dict


if __name__ == "__main__":
	tck = TopColorKMeans()
	img = cv2.imread('./img1.jpg')
	pct, hex_color, color_variation = tck.get_top_color(img)

	print('Top Color:\t', hex_color)
	#print('Percentage:\t', [f'{round(p*100, 2)}%' for p in pct])
	print('\nPossible Color Variation from the above color (not including the original color above):')
	print('Complementary Color: ', color_variation['var_2'])
	print('Triadic Color: ', color_variation['var_3'])
	print('Tetradic Color: ', color_variation['var_4'])