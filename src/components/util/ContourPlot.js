
import * as tf from '@tensorflow/tfjs'
import * as d3 from 'd3';

export class ContourPlot {
    constructor(svg, coordinateConversions) {
        this.svg = svg
        this.coordinateConversions = coordinateConversions

        this.nx = 256
        this.ny = 256
    }

    prerender(losses) {
        this.colorContours = losses.map(loss => {
            var lossGrid = this.getLossGrid(loss)
            var contours = this.getContours(lossGrid)
            var colorScale = this.getColorScale(lossGrid)

            return contours.map(c => {
                return { ...c, value: colorScale(c.value) }
            })
        })
    }

    render(i) {
        var colorContours = this.colorContours[i]

        this.plot = this.contourAttributes(
            this.svg.append("g")
                .attr("fill", "none")
                .attr("stroke", "#fff")
                .attr("stroke-opacity", 0.3)
                .selectAll("path")
                .data(colorContours)
                .enter()
                .append("path")
        )
        this.plot.exit().remove()
    }

    rerender(i) {
        var colorContours = this.colorContours[i]

        this.plot = this.contourAttributes(this.plot.data([]))
        this.plot.exit().remove()

        this.plot = this.contourAttributes(this.plot.data(colorContours).enter().append("path"))
    }

    getContours(lossGrid) {
        const n_thresholds = 15;

        return d3.contours()
            .size([this.nx, this.ny])
            .thresholds(n_thresholds)
            (lossGrid)
    }

    getLossGrid(loss) {
        var grid = tf.stack(tf.meshgrid(
            tf.linspace(...this.coordinateConversions.xrange, this.nx),
            tf.linspace(...this.coordinateConversions.yrange, this.ny)
        ), -1)

        return Array.from(loss(grid).bufferSync().values)
    }

    getColorScale(lossGrid) {
        return d3.scaleLinear().domain(d3.extent(lossGrid)).range([0, 1])
    }

    contourAttributes = (binding) => binding
        .attr("fill", d => d3.interpolateViridis(d.value))
        .attr("stroke", "#white")
        .attr("transform", `translate(0,${this.coordinateConversions.innerHeight}), scale(${(this.coordinateConversions.innerWidth / this.nx)}, ${(-this.coordinateConversions.innerHeight / this.ny)})`)
        .attr("d", d3.geoPath(d3.geoIdentity()))
}