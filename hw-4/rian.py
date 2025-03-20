def curve(pixel):
    r, g, b = pixel
    brightness = r + g + b
    if brightness < 60:
        k = 60 / brightness
        return min(255, int(r * k ** 2)), min(255, int(g * k ** 2)), min(255, int(b* k
** 2))
    else:
        return r, g, b

from PIL import Image
im = Image.open("im1.jpg")
pixels = im.load() # список с пикселями
x, y = im.size # ширина (x) и высота (y) изображения
print(x, y)
n = min(x,y)
for i in range(n):
    for j in range(i+1, n):
        r, g, b = pixels[i, j]
        pixels[i, j], pixels[j, i] =  pixels[j, i], pixels[i, j]
im.save("Риана2.jpg")
