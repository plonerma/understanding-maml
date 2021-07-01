
import * as tf from '@tensorflow/tfjs'


export class MetaVariable {
    constructor(initialValues) {
        this.state = Array.isArray(initialValues) ? tf.tensor(initialValues) : initialValues
    }

    transform(model) {
        return model.transform(this.state)
    }

    toArray() {
        return this.state.bufferSync().values
    }
}

/**
 * Implementation of Reptile
 */
export class Reptile {
    constructor(metaInterpolationRate, innerLearningRate, nSteps = 1) {
        this.metaInterpolationRate = metaInterpolationRate
        this.innerLearningRate = innerLearningRate
        this.nSteps = nSteps
    }

    transform(state) {
        let params = state.clone()
        return lossGradients => new MetaVariable(this.updateParameters(lossGradients, params))
    }

    updateParameters(lossGradients, params) {
        for (let i = 0; i < lossGradients.length; i++) {
            let optimalInnerParams = vanillaGradientDescent(lossGradients[i], params, this.innerLearningRate, this.nSteps)
            params = params.add(optimalInnerParams.sub(params).mul(this.metaInterpolationRate))
        }
        return params
    }
}

/**
 * Implementation of first-order MAML.
 */
export class FirstOrderMAML {
    constructor(metaLearningRate) {
        this.metaLearningRate = metaLearningRate
    }

    transform(state) {
        let params = state.clone()
        return (lossGradients, innerUpdates) => new MetaVariable(
            this.updateParameters(lossGradients, innerUpdates, params))
    }

    updateParameters(lossGradients, innerUpdates, params) {
        let taskLossGradients = lossGradients.map(
            (lossGradient, i) => lossGradient([innerUpdates[i]]))
        return params.sub(tf.sum(tf.stack(taskLossGradients), 0).mul(this.metaLearningRate))
    }
}

/**
 * Performs nSteps steps of vanilla gradient descent on a provided loss, given some initial params and a learning rate.
 * @param {function} gradient Gradient of the loss. Obtained from tf.grad(loss).
 * @param {tf.Tensor} params Initial parameters.
 * @param {float} learningRate Static learning rate.
 * @param {int} nSteps Number of steps.
 * @returns The updated parameters.
 */
const vanillaGradientDescent = (gradient, params, learningRate, nSteps) => {
    return tf.tidy(() => {
        for (let _ = 0; _ < nSteps; _++) {
            params = params.sub(gradient(params).mul(learningRate))
        }
        return params
    })
}