
import * as d3 from 'd3';

export class CoordinateConversions {
    constructor(margin, width, height, xrange, yrange) {
        this.innerWidth = width - margin.left - margin.right
        this.innerHeight = height - margin.top - margin.bottom

        this.xrange = xrange 
        this.yrange = yrange 

        this.margin = margin 
        this.width = width 
        this.height = height
    }

    onHorizontalAxis = (x) => {
        var relativePosition = (x - this.xrange[0]) / (this.xrange[1] - this.xrange[0])
        return relativePosition * this.innerWidth
    }
    
    fromHorizontalAxis = (x) => {
        var absolutePosition = x / this.innerWidth 
        return absolutePosition * (this.xrange[1] - this.xrange[0]) + this.xrange[0]
    }
    
    onVerticalAxis = (y) => {
        var relativePosition = (y - this.yrange[0]) / (this.yrange[1] - this.yrange[0])
        return this.innerHeight - relativePosition * this.innerHeight
    }
    
    fromVerticalAxis = (y) => {
        var absolutePosition = (this.innerHeight - y) / this.innerHeight
        return absolutePosition * (this.yrange[1] - this.yrange[0]) + this.xrange[0]
    }
    
    scaleVectorComponent = (z1, z2, by) => (z2 - z1) * by + z1  

    d3eventToCoordinates = (event) => {
        return {
            x: this.fromHorizontalAxis(event.x),
            y: this.fromVerticalAxis(event.y)
        }
    }

    d3sourceEventToCoordinates = (event) => {
        return {
            x: this.fromHorizontalAxis(event.sourceEvent.clientX) + this.margin.left,
            y: this.fromVerticalAxis(event.sourceEvent.clientY) + this.margin.top
        }
    }
}