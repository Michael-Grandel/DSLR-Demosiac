# DSLR-Demosiac
3 simple algorithms that compiles an image from DSLR camera inputs.

This program is done via jupyter notebook. You can change the images tested in the following code snippets:


__This part is for the comparison of mosiac and truth image__

`f, axarr = plt.subplots(1,2)`

`axarr[0].imshow(mosaiced_images[0], cmap='gray') #Change the index for new images `

`axarr[1].imshow(gtruth_images[0], cmap='gray') #Change the index for new images `


__This part is for the demosiac image output (at the end of the code)__

`image_to_show = 2 #Change the index for new demosiac image`
