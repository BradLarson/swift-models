# Physarum transport model

This implements a version of the [Physarum transport model](https://www.sagejenson.com/physarum)
described by Sage Jenson, inspired by [“Characteristics of pattern formation and evolution in 
approximations of Physarum transport networks.”](http://eprints.uwe.ac.uk/15260/1/artl.2010.16.2.pdf)

Representative animated images of the evolution of the world state will be written into `output/`.

## Setup

To begin, you'll need the [latest version of Swift for
TensorFlow](https://github.com/tensorflow/swift/blob/master/Installation.md)
installed. Make sure you've added the correct version of `swift` to your path.

To run the model, use the following:

```sh
cd swift-models
swift run -c release Physarum
```
