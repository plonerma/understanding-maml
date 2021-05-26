
import * as tf from '@tensorflow/tfjs'

export class Random2DLinearRegressionLossSpace {
    
    /**
     * Generate a random 2D linear regression task and compute its loss space w.r.t to two free parameters a and b in:
     * 
     * y = ax + b
     * 
     * where loss({a, b}) = .5 * (y_true - ax - b)**2, summed over all samples. Note that the input is 1D but the parameter space is 2D.
     */
    constructor() {
        this.trueParameters = tf.randomUniform([2])
        this.trueParameters.print()
        
        var x = tf.randomUniform([10])
        var z = tf.stack([x, tf.ones([10])])

        z.print() 
        var y = tf.dot(this.trueParameters, z)

        var loss = parameters => (y.sub(tf.dot(parameters, z))).square().mul(tf.scalar(.5))

        var lossGrad = tf.grad(loss)
        
        lossGrad(tf.tensor1d([.5, .5])).print()
    }
}