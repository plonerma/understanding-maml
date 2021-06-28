
import * as tf from '@tensorflow/tfjs'

export const sgd = () => {
    this.a = tf.scalar(-1).variable()
    this.b = tf.scalar(1).variable()

    console.log(
        `a: ${this.a.dataSync()}, b: ${this.b.dataSync()}`)

    var N = 100

    var xs = tf.randomUniform([N])
    //var z = tf.stack([x, tf.ones([N])])
    var ys = xs.mul(.5).add(.5)

    const f = x => x.mul(this.a).add(this.b)

    const loss = (pred, label) => pred.sub(label).square().mean()

    const learningRate = 0.9
    const optimizer = tf.train.sgd(learningRate)

    // Train the model.
    for (let i = 0; i < 10; i++) {
        optimizer.minimize(() => loss(f(xs), ys))
        console.log(
            `a: ${this.a.dataSync()}, b: ${this.b.dataSync()}`)
    }   
}

import { sampleIndependentMultivariateGaussian } from './Sampling.js'

export class Random2DLinearRegressionLossSpace {

    /**
     * Generate a random 2D linear regression task and compute its loss space w.r.t to two free parameters a and b in:
     * 
     * y = ax + b
     * 
     * where loss({a, b}) = .5 * (y_true - ax - b)**2, summed over all samples. Note that the input is 1D but the parameter space is 2D.
     */
    constructor(mean = [.5, .5], variance = [.01, .01]) {
        this.trueParameters = sampleIndependentMultivariateGaussian(mean, variance)
        this.trueParameters.print()
        var N = 10
        var x = tf.randomUniform([N])
        var z = tf.stack([x, tf.ones([N])])
        var y = tf.dot(this.trueParameters, z)

        this.loss = tf.tidy(() => parameters => {
            var estimates = tf.matMul(parameters, z)
            var squaredError = estimates.squaredDifference(y)
            var sumOfSquares = squaredError.sum(-1)
            return sumOfSquares.div(tf.scalar(2 * N))
        })

        this.lossGrad = tf.grad(this.loss)
    }

    paramUpdate(input, lr = 0.05, makeTensor = false) {
        if(makeTensor){
            input = tf.tensor(input)
        }
        
        lr = tf.scalar(lr)
        return tf.tidy(() => input.sub(lr.mul(this.lossGrad(input))))
    }

    toArray(tensor) {
        return tensor.bufferSync().values
    }
}