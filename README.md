# googLeNet
here is my implementation of the basic googLeNet described in the paper [Going Deeper with Convolutions](http://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html) in keras

I wrapped the inception module described in the paper and put it in the file "inceptionModel.py"

the structure of the net is described as following:

![pic](structure.png)

### MODIFIED:

#### googleNet_add_deep.py:

following inception block 5, add an extra inception block named with 6

| type     | patch size/stride | output size | 1\*1 | 3\*3reduce | 3\*3 | 5\*5redude | 5\*5 | pool proj |
| -------- | ----------------- | ----------- | ---- | ---------- | ---- | ---------- | ---- | --------- |
| max pool | 3\*3/2            | 8\*8\*1024  |      |            |      |            |      |           |
| incep_6a |                   | 8\*8\*1024  | 384  | 192        | 384  | 48         | 128  | 128       |
| incep_6b |                   | 8\*8\*1408  | 512  | 256        | 512  | 48         | 192  | 192       |
| avg pool | 8\*8/1            | 1\*1\*1408  |      |            |      |            |      |           |

#### googleNet_pool.py:

add an extra pooling layer in the last avg pooling layer

| type      | patch size/stride | output size |
| --------- | ----------------- | ----------- |
| max pool  | 3*3/2             | 8\*8\*1024  |
| avg  pool | 8\*8/1            | 1\*1\*1024  |