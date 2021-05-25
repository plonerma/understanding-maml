
import * as d3 from 'd3';

export class VectorGroup {
    constructor(svg, origin, vectors, coordinateConversions) {
        this.svg = svg
        this.origin = origin
        this.vectors = vectors
        this.coordinateConversions = coordinateConversions

        this.defaultOrigin = origin
    }

    declare() {
        this.positionLabels()
        this.renderVectorsWithShades()
        this.renderOrigin()

        return this 
    }

    animate() {
        this.animateOriginDrag()
        return
        this.animateVectorDrag()

        this.vectorShadeElements.on("mouseover", (event) => {
            d3.select(event.target)
                .attr("stroke-opacity", 0.4)
        })

        this.vectorShadeElements.on("mouseout", (event) => {
            d3.select(event.target)
                .attr("stroke-opacity", 0.0)
        })

        return this
    }

    positionLabels() {
        /*this.vectors.forEach(vector => this.labelAttributes(
            d3.select(`#${vector.id}`)
                .datum(vector)
        ))*/
    }

    renderVectorsWithShades() {
        var vectorData = this.vectorsWithOriginData()

        this.vectorMarkerAttributes(
            this.svg.selectAll("defs")
                .data(vectorData)
                .enter().append("defs").append("marker")
        )

        this.vectorElements = this.vectorAttributes(
            this.svg.append("g")
                .selectAll("line")
                .data(vectorData)
                .enter().append("line")
        )

        this.vectorShadeElements = this.vectorShadeAttributes(
            this.svg.append("g")
                .selectAll("line")
                .data(vectorData)
                .enter().append("line")
        )
    }

    renderOrigin() {
        console.log(this.origin)
        this.originElement = this.originAttributes(
            this.svg.selectAll("circle")
                .data([this.origin]).enter()
                .append("circle")
        )
    }

    animateOriginDrag() {
        d3.drag().on("drag", (event) => {
            var [x,y] = d3.pointer(event, this.svg.node())
            x = xscale.invert(x)
            y = yscale.invert(y)
            x = Math.max(Math.min(1, x), 0)
            y = Math.max(Math.min(1, y), 0)
            console.log(x, y)
            this.origin = this.coordinateConversions.d3sourceEventToCoordinates(event)
            var vectorData = this.vectorsWithOriginData()

            this.originAttributes(this.originElement.datum(this.origin))
            this.vectorAttributes(this.vectorElements.data(vectorData))
            this.vectorShadeAttributes(this.vectorShadeElements.data(vectorData))

        })(this.originElement)
    }

    animateVectorDrag() {
        d3.drag().on("drag", (event) => {
            this.vectors = this.vectors.map(vector => vector.id == event.subject.id ? {
                ...vector,
                ...this.coordinateConversions.d3sourceEventToCoordinates(event)
            } : vector)

            var vectorData = this.vectorsWithOriginData()

            this.positionLabels()
            this.vectorAttributes(this.vectorElements.data(vectorData))
            this.vectorShadeAttributes(this.vectorShadeElements.data(vectorData))
        })(this.vectorShadeElements)
    }

    vectorsWithOriginData() {
        return this.vectors.map(vector => {
            return { ...vector, origin: this.origin }
        })
    } 

    /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Styling!
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
    vectorMarkerAttributes = (binding) => binding
        .attr("id", d => `density-marker-${d.id}`)
        .attr("markerWidth", 3)
        .attr("markerHeight", 3)
        .attr("refX", 0)
        .attr("refY", 1.5)
        .attr("orient", "auto")
        .append("polygon")
        .attr("points", "0 0, 3 1.5, 0 3")

    vectorAttributes = (binding) => binding
        .attr("x1", d => this.coordinateConversions.onHorizontalAxis(d.origin.x))
        .attr("x2", d => this.coordinateConversions.onHorizontalAxis(d.x))
        .attr("y1", d => this.coordinateConversions.onVerticalAxis(d.origin.y))
        .attr("y2", d => this.coordinateConversions.onVerticalAxis(d.y))
        .attr("stroke", "#000")
        .attr("stroke-width", 2)
        .attr("marker-end", d => `url(#density-marker-${d.id})`)
        .attr("pointer-events", "bounding-box")
        .attr("cursor", "pointer")

    vectorShadeAttributes = (binding) => binding
        .attr("x1", d => this.coordinateConversions.onHorizontalAxis(d.origin.x))
        .attr("x2", d => this.coordinateConversions.onHorizontalAxis(this.coordinateConversions.scaleVectorComponent(d.origin.x, d.x, 1.05)))
        .attr("y1", d => this.coordinateConversions.onVerticalAxis(d.origin.y))
        .attr("y2", d => this.coordinateConversions.onVerticalAxis(this.coordinateConversions.scaleVectorComponent(d.origin.y, d.y, 1.05)))
        .attr("stroke", "lightgreen")
        .attr("stroke-width", 16)
        .attr("stroke-opacity", 0.0)
        .attr("cursor", "pointer")

    originAttributes = (binding) => binding
        .attr("cx", d => this.coordinateConversions.onHorizontalAxis(d.x))
        .attr("cy", d => this.coordinateConversions.onVerticalAxis(d.y))
        .attr("r", 4)

    labelAttributes = (binding) => binding
        .attr("style", d => `
        position: absolute; 
        left: ${this.coordinateConversions.onHorizontalAxis(d.x) + 15 + "px"};
        top: ${this.coordinateConversions.onVerticalAxis(d.y) + 30 + "px"};
    `)
}