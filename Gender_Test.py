import numpy as np
from keras import losses, optimizers, models
from keras import utils
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

# Get Image-Path
img_name = input("Was ist der Bild Name?")
path = "TestPictures/" + img_name + ".jpg"

# Shape
img_width = 200
img_height = 200

# Load and Compile Model
model = models.load_model("Gender.h5")
model.compile(loss=losses.binary_crossentropy, optimizer=optimizers.Adadelta(), metrics=['acc'])

# Get Image
img = utils.load_img(path, target_size=(img_width, img_height))
x = utils.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Get
pred = model.predict(x)
print(pred)
if pred > 0.5:
    pred = "Male"
else:
    pred = "Female"

resizedpath = "ResizedPictures/" + img_name + ".jpg"
image = Image.open(path)
new_image = image.resize((200, 200))
new_image.save(resizedpath)

plt.title(pred)
plt.xlabel("X-Pixel")
plt.ylabel("Y-Pixel")

image = mpimg.imread(resizedpath)
image.resize()

plt.imshow(image)
plt.show()
