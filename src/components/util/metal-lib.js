import * as tf from '@tensorflow/tfjs'

export class Model {

    /**
     * Wrapper Model superclass that conveniently calls the parameter updates.
     * @param {Array|tf.Tensor} variable Either an array of data or already a tensor.
     * @param  {...any} args Any args to apply to the updateParameters-call.
     * @returns The result of the parameter update.
     */
    update(variable, ...args) {
        if (Array.isArray(variable)) {
            let params = tf.tensor(variable)
            return tf.tidy(() => this.updateParameters(params, ...args))
        }
        return tf.tidy(() => this.updateParameters(variable, ...args))
    }

    updateParameters() {
        throw new Error("This model does not implement the 'updateParameters' method.")
    }
}

/**
 * Since tfjs does not provide a tf.jacobian, we use finite differences to estimate the Jacobian.
 * This isn't optimal, but required for MAML.
 * @param {function} f Vector-valued function.
 * @returns function that computes the Jacobian J(x).
 */
const jacobian = (f) => {
    const sliceRow = (tensor, rowIndex) => tensor.slice([rowIndex, 0], [1, tensor.shape[1]])
    const perturbationCoeff = 0.01
    return (x) => {
        const perturb = tf.eye(x.shape[1]).mul(perturbationCoeff)
        const jac = tf.stack(x.shape.map((_, i) => f(x.add(sliceRow(perturb, i))).sub(f(x)).div(perturbationCoeff)))
        return jac.squeeze()
    }
}

export class MAML extends Model {

    /**
     * Implements Finn et al. (2017) MAML update step.
     * @param {float} metaLearningRate meta learning rate
     * @param {float} innerLearningRate inner learning rate
     * @param {int} nSteps Number of inner gradient descent steps
     */
    constructor(metaLearningRate, innerLearningRate, nSteps = 1) {
        super()
        this.metaLearningRate = metaLearningRate
        this.innerLearningRate = innerLearningRate
        this.vGD = new VanillaGradientDescent(innerLearningRate, nSteps)
    }

    /**
     * Performs a MAML step given a fixed set of loss gradients (each obtained from tf.grad(loss)), where loss is the loss-function of 
     * the respective task.
     * @param {Array<function>} lossGradients List of loss gradients. Obtained from tf.grad(loss).
     * @param {tf.Tensor} params Parameters to be updated. 
     * @returns Updated parameters.
     */
    updateParameters(params, lossGradients) {
        let optimalInnerUpdates = lossGradients.map(lossGradient => this.vGD.update(params, lossGradient))
        let taskLossGradients = lossGradients.map(
            (lossGradient, i) => {
                let gradientOptimalInner = lossGradient(optimalInnerUpdates[i])
                let taskLossJacobian = jacobian(lossGradients[i])(params).mul(this.innerLearningRate)
                return gradientOptimalInner.sub(tf.matMul(gradientOptimalInner, taskLossJacobian))
            })

        return params.sub(tf.sum(tf.stack(taskLossGradients), 0).mul(this.metaLearningRate))
    }
}

export class Reptile extends Model {

    /**
     * Implements the Nicho et al. (2018) Reptile update step. 
     * @param {float} metaInterpolationRate interpolation rate between old and new parameters
     * @param {float} innerLearningRate inner learning rate
     * @param {int} nSteps inner gradient descent steps 
     */
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
    }
}

export class FirstOrderMAML extends MAML {

    /**
     * Performs a first-order MAML step given a fixed set of loss gradients (each obtained from tf.grad(loss)), where loss is the loss-function of 
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

export class iMAML extends MAML {

    /**
     * Performs a first-order MAML step given a fixed set of loss gradients (each obtained from tf.grad(loss)), where loss is the loss-function of 
     * the respective task.
     * @param {Array<function>} lossGradients List of loss gradients. Obtained from tf.grad(loss).
     * @param {tf.Tensor} params Parameters to be updated. 
     * @returns Updated parameters.
     */
    updateParameters(params, lossGradients) {
        // TODO: @plonerma
        return params
    }
}

export class VanillaGradientDescent extends Model {
    /**
     * Implements vanilla gradient descent
     * @param {float} learningRate Static learning rate.
     * @param {int} nSteps Number of steps.
     */
    constructor(learningRate, nSteps, returnAllParams = false) {
        super()
        this.learningRate = learningRate
        this.nSteps = nSteps
        this.returnAllParams = returnAllParams
    }

    /**
     * Performs nSteps steps of vanilla gradient descent on a provided loss, given some initial params and a learning rate.
     * @param {tf.Tensor} params Initial parameters.
     * @param {function} gradient Gradient of the loss. Obtained from tf.grad(loss).
     * @returns The updated parameters.
     */
    updateParameters(params, gradient) {
        let paramList = [ params ]
        for (let i = 0; i < this.nSteps; i++) {
            paramList.push(
                paramList[i].sub(gradient(paramList[i]).mul(this.learningRate))
            )
        }
        return this.returnAllParams ? paramList : paramList[paramList.length - 1]
    }
}



export class IMAML extends Model {

    /**
     * Implements the Rajeswaran et al. (2019) iMAML update step.
     * @param {float} metaLearningRate outer learning rate
     * @param {float} regularizationCoefficient regularization coeffiecient
     * @param {float} innerLearningRate inner learning rate
     * @param {int} nSteps inner gradient descent steps
     */
    constructor(metaLearningRate, innerLearningRate, nSteps = 10) {
        super()
        this.metaLearningRate = metaLearningRate
        this.regularizationCoefficient = regularizationCoefficient
        this.vGD = new VanillaGradientDescent(innerLearningRate, nSteps)
    }

    /**
     * Performs an iMAML step given a fixed set of loss gradients (each obtained
     * from tf.grad(loss)), where loss is the loss-function of the respective task.
     * @param {Array<function>} lossGradients List of loss gradients. Obtained
     *                                        from tf.grad(loss).
     * @param {tf.Tensor} params Parameters to be updated.
     * @returns Updated parameters.
     */
    updateParameters(params, lossGradients) {
        let optimalInnerUpdates = lossGradients.map(lossGradient => this.vGD.update(params, lossGradient))

        return optimalInnerUpdates.reduce((aggregatedParams, optimalInnerParams) => {
            return aggregatedParams.add(optimalInnerParams.sub(aggregatedParams).mul(this.metaInterpolationRate))
        }, params)
    }
}
