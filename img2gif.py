import imageio
import glob

images = []

location = 'gifpictures/*.png'
savegifs = 'gifs/test.gif'

def makegif(location, savegifs):
	for img in sorted(glob.glob(location)):
		images.append(imageio.imread(img))

	imageio.mimsave(savegifs, images)


if __name__ == '__main__':
	makegif(location, savegifs)

