使用奇异值分解（SVD）实现了图像压缩算法。这是学习线性代数时给自己布置的作业

```
usage: svd.py [-h] [-c FILE] [-o FILE] [-r RATE] [-d FILE] [-p FILE]
              [--rates RATES [RATES ...]]

SVD Image Compression

options:
  -h, --help            show this help message and exit
  -c FILE, --compress FILE
                        Compress an image
  -o FILE, --output FILE
                        Specify output file for compression
  -r RATE, --rate RATE  Compression rate
  -d FILE, --decompress FILE
                        Decompress a compressed file
  -p FILE, --preview FILE
                        Preview the compressed images
  --rates RATES [RATES ...]
                        Specify compression rates for preview
```