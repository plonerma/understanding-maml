
import * as tf from '@tensorflow/tfjs'
import * as d3 from 'd3';

export class ContourPlot {
    constructor(loss, svg, coordinateConversions){
        this.loss = loss
        this.svg = svg
        this.coordinateConversions = coordinateConversions
    }

    render() {
        const nx = 256, ny = 256;
        const n_thresholds = 20;

        var grid = tf.stack(tf.meshgrid(
            tf.linspace(...this.coordinateConversions.xrange, nx),
            tf.linspace(...this.coordinateConversions.yrange, ny)
        ), -1)

        var lossGrid = Array.from(tf.log(this.loss(grid)).bufferSync().values)

        var contours = d3.contours()
            .size([nx, ny])
            .thresholds(n_thresholds)
            (lossGrid)

        var color_map = d3.interpolateViridis
        var color_scale = d3.scaleLinear().domain(d3.extent(lossGrid)).range([0, 1])
        
        this.svg.append("g")
            .attr("fill", "none")
            .attr("stroke", "#fff")
            .attr("stroke-opacity", 0.3)
            .selectAll("path")
            .data(contours)
            .enter()
            .append("path")
            .attr("fill", d => color_map(color_scale(d.value)))
            .attr("stroke", "#white")
            .attr("transform", `translate(0,${this.coordinateConversions.innerHeight}), scale(${(this.coordinateConversions.innerWidth / nx)}, ${(-this.coordinateConversions.innerHeight / ny)})`)
            .attr("d", d3.geoPath(d3.geoIdentity()))
    }
}