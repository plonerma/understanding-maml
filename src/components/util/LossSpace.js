
import * as tf from '@tensorflow/tfjs'
import { square } from '@tensorflow/tfjs'

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
    constructor(mean = [.5, .5], variance = [.00, .00]) {
        this.trueParameters = tf.tensor1d(mean)//sampleIndependentMultivariateGaussian(mean, variance)
        this.trueParameters.print()
        var N = 10
        var x = tf.randomUniform([N], -1, 1)
        var z = tf.stack([x, tf.ones([N])])
        var y = tf.dot(this.trueParameters, z).elu()

        x.print()

        this.loss = tf.tidy(() => parameters => {
            var estimates = tf.matMul(parameters, z).elu()
            var squaredError = estimates.squaredDifference(y)
            var sumOfSquares = squaredError.sum(-1)
            return sumOfSquares.div(tf.scalar(2))
        })

        this.lossGrad = tf.grad(this.loss)
    }

    paramUpdate(input, lr = 0.05, makeTensor = false, nSteps = 1) {
        if(makeTensor){
            input = tf.tensor(input)
        }
        
        lr = tf.scalar(lr)
        return tf.tidy(() => {
            var param = input
            for(var i = 0; i < nSteps; i++){
                param = param.sub(lr.mul(this.lossGrad(param)))
            }
            return param
        })
    }

    toArray(tensor) {
        return tensor.bufferSync().values
    }
}

export class CurvatureLoss {

    constructor(curvature = 1) {
        this.trueParameters = tf.tensor1d([.5, .5])

        this.loss = tf.tidy(() => parameters => {

            let diffVec = parameters.sub(this.trueParameters)
            let f = tf.einsum('ijk,kji->ij', diffVec.matMul(tf.diag([.1, 1]).mul(curvature)), diffVec.transpose())
            return f.div(f.add(.01))
        })

        this.lossGrad = tf.grad(this.loss)
    }

    toArray(tensor) {
        return tensor.bufferSync().values
    }
}