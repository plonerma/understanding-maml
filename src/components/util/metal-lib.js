
import * as tf from '@tensorflow/tfjs'


export class MetaVariable {
    constructor(initialValues) {
        this.state = Array.isArray(initialValues) ? tf.tensor(initialValues) : initialValues
    }

    getStateCopy() {
        return this.state.clone()
    }

    toArray() {
        return this.state.bufferSync().values
    }
}

class Model {
    update(variable, ...args) {
        if(variable instanceof MetaVariable){
            let params = variable.getStateCopy()
            return new MetaVariable(tf.tidy(() => this.updateParameters(params, ...args)))
        }
        return tf.tidy(() => this.updateParameters(variable, ...args))
    }

    updateParameters() {
        throw new Error("This model does not implement the 'updateParameters' method.")
    }
}

/**
 * Implementation of Reptile
 */
export class Reptile extends Model {
    constructor(metaInterpolationRate, innerLearningRate, nSteps = 1) {
        super()
        this.metaInterpolationRate = metaInterpolationRate
        this.vGD = new VanillaGradientDescent(innerLearningRate, nSteps)
    }

    /**
     * Performs a Reptile step given a fixed set of loss gradients (each obtained from tf.grad(loss)), where loss is the loss-function of 
     * the respective task.
     * @param {Array<function>} lossGradients List of loss gradients. Obtained from tf.grad(loss).
     * @param {tf.Tensor} params Parameters to be updated. 
     * @returns Updated parameters.
     */
    updateParameters(params, lossGradients) {
        let optimalInnerUpdates = lossGradients.map(lossGradient => this.vGD.update(params, lossGradient))
        
        return optimalInnerUpdates.reduce((aggregatedParams, optimalInnerParams) => {
            return aggregatedParams.add(optimalInnerParams.sub(aggregatedParams).mul(this.metaInterpolationRate))
        }, params)
        
        for (let i = 0; i < lossGradients.length; i++) {
            let optimalInnerParams = this.vGD.update(params, lossGradients[i])
            params = params.add(optimalInnerParams.sub(params).mul(this.metaInterpolationRate))
        }
        return params
    }
}

export class FirstOrderMAML extends Model {

    /**
     * Implementation of first-order MAML.
     * @param {float} metaLearningRate 
     */
    constructor(metaLearningRate, innerLearningRate, nSteps = 1) {
        super()
        this.metaLearningRate = metaLearningRate
        this.vGD = new VanillaGradientDescent(innerLearningRate, nSteps)
    }

    /**
     * Performs a Reptile step given a fixed set of loss gradients (each obtained from tf.grad(loss)), where loss is the loss-function of 
     * the respective task.
     * @param {Array<function>} lossGradients List of loss gradients. Obtained from tf.grad(loss).
     * @param {tf.Tensor} params Parameters to be updated. 
     * @returns Updated parameters.
     */
    updateParameters(params, lossGradients) {
        let optimalInnerUpdates = lossGradients.map(lossGradient => this.vGD.update(params, lossGradient))
        let taskLossGradients = lossGradients.map(
            (lossGradient, i) => lossGradient(optimalInnerUpdates[i]))
        return params.sub(tf.sum(tf.stack(taskLossGradients), 0).mul(this.metaLearningRate))
    }
}

export class VanillaGradientDescent extends Model {
    /**
     * Implements vanilla gradient descent
     * @param {float} learningRate Static learning rate.
     * @param {int} nSteps Number of steps.
     */
    constructor(learningRate, nSteps) {
        super()
        this.learningRate = learningRate
        this.nSteps = nSteps
    }

    /**
     * Performs nSteps steps of vanilla gradient descent on a provided loss, given some initial params and a learning rate.
     * @param {tf.Tensor} params Initial parameters.
     * @param {function} gradient Gradient of the loss. Obtained from tf.grad(loss).
     * @returns The updated parameters.
     */
    updateParameters(params, gradient) {
        for (let _ = 0; _ < this.nSteps; _++) {
            params = params.sub(gradient(params).mul(this.learningRate))
        }
        return params
    }
}