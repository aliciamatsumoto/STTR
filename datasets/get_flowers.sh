#!/bin/bash
curl -OL https://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz
tar -xzvf 17flowers.tgz
rm 17flowers.tgz
mv jpg flowers