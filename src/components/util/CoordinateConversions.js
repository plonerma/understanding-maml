
import * as d3 from 'd3';

export class CoordinateConversions {
    constructor(g, margin, width, height, xrange, yrange) {
        this.g = g

        this.innerWidth = width - margin.left - margin.right
        this.innerHeight = height - margin.top - margin.bottom

        this.xrange = xrange
        this.yrange = yrange

        this.margin = margin
        this.width = width
        this.height = height

        this.xscale = d3.scaleLinear().domain(xrange).range([0, this.innerWidth])
        this.yscale = d3.scaleLinear().domain(yrange).range([this.innerHeight, 0])
    }

    displayAxes() {
        this.g.append("g").attr("transform", "translate(0," + this.innerHeight + ")").call(d3.axisBottom(this.xscale))
        this.g.append("g").call(d3.axisLeft(this.yscale))
    }

    onHorizontalAxis = (x) => {
        return this.xscale(x)
    }

    fromHorizontalAxis = (x) => {
        var absolutePosition = x / this.innerWidth
        return absolutePosition * (this.xrange[1] - this.xrange[0]) + this.xrange[0]
    }

    onVerticalAxis = (y) => {
        return this.yscale(y)
    }

    fromVerticalAxis = (y) => {
        var absolutePosition = (this.innerHeight - y) / this.innerHeight
        return absolutePosition * (this.yrange[1] - this.yrange[0]) + this.xrange[0]
    }

    scaleVectorComponent = (z1, z2, by) => (z2 - z1) * by + z1

    scaleVectorComponent2 = (z1, z2, by) => (z2 - z1) * by + z1

    vectorLength = (x1, x2, y1, y2) => Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    offsetVectorComponent = (z1, z2, w, l) => {
        return (z2 - z1) * l / w + z2
    }

    offsetVectorComponent2 = (z1, z2, by) => z2 + by 

    d3eventToCoordinates = (event) => {
        var [x, y] = d3.pointer(event, this.g.node())
        x = this.xscale.invert(x)
        y = this.yscale.invert(y)
        x = Math.max(Math.min(this.xrange[1], x), this.xrange[0])
        y = Math.max(Math.min(this.yrange[1], y), this.yrange[0])
        return { x: x, y: y}
    }

    d3sourceEventToCoordinates = (event) => {
        return {
            x: this.fromHorizontalAxis(event.sourceEvent.layerX - this.margin.left),
            y: this.fromVerticalAxis(event.sourceEvent.layerY - this.margin.top)
        }
    }


}