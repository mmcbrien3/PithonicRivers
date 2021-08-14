## Pithonic Rivers

This code is used for determining the "sinuosity" of rivers. The aim of this
code is to take in satellite images of rivers from the real world and determine
how curvy the river is. 

The sinuosity of a river can be calculated with the following:
```
Let d = distance along river from source to mouth
Let c = distsance from source to mouth of river in a straight line
sinuosity = d/c
```

Therefore, a straight river would have a sinuosity of exactly 1 since in that
case `c=d`.

This project was inspired by the following numberphile video about the
sinuosity of rivers: 
[link to numberphile video](https://www.youtube.com/watch?v=TUErNWBOkUM)

[In a follow-up article](https://www.theguardian.com/science/alexs-adventures-in-numberland/2015/mar/14/pi-day-2015-pi-rivers-truth-grime),
the mathematician from the numberphile video, 
James Grime, conducted a crowd-sourcing effort to determine the actual average 
sinuosity of rivers around the world. This resulted in a value of 1.94, 
not the magical Pi suggested by the original paper.

A fun fact stated from the guardian article: `1.94 = Pi / Golden Ratio`