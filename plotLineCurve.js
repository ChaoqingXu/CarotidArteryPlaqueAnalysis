
    // set the dimensions and margins of the graph
    var margin = { top: 10, right: 100, bottom: 30, left: 30 },
        width = 860 - margin.left - margin.right,
        height = 800 - margin.top - margin.bottom;

    // append the svg object to the body of the page
    var svg = d3.select("#my_dataviz")
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");

    //Read the data
    // d3.csv("https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/data_connectedscatter.csv", function (data) {
    d3.csv("https://raw.githubusercontent.com/ChaoqingXu/CarotidArteryPlaqueAnalysis/main/CarotidArteryData_HTML/HTMLPlotCSV.csv", function (data) {

        width_value = 400  // x-axis 
        height_value = 1 // since y-values are normalized

        
        var subject_ID_labels = data.columns
        var Indexlist = subject_ID_labels.shift() // remove the first column and store it in Indexlist

        // Reformat the data: we need an array of arrays of {x, y} tuples
        var dataReady = subject_ID_labels.map(function (grpName) { // .map allows to do something for each element of the list
            return {
                name: grpName,
                values: data.map(function (d) {
                    return { subject_label: d.subject_label, value: +d[grpName] };
                })
            };
        });
        // I strongly advise to have a look to dataReady with
        console.log(dataReady)

        // match subject labels categories, based on subject_ID_labels last string
        var labels = ['Calcium', 'Fibrous', 'IPH_lipid', 'IPH']
        var category = []
        const match = subject_ID_labels.find(element => {
            for (let i = 0; i < labels.length; i++) {
                key = labels[i];
                if (key !== 'IPH') {
                    if (element.includes(key)) {
                        category.push(key);
                    }
                }
                if (key == 'IPH') {
                    if (element.includes(key) && element.includes('IPH_lipid')) {
                        category.push(key);
                    }
                }
            }
        });
        // console.log(category)

        
        // A color scale: one color for each group
        var myColor = d3.scaleOrdinal()
            .domain(subject_ID_labels)
            .range(d3.schemeSet2);
            
        // Color scale: give me a specie name, I return a color
        var labelColor = d3.scaleOrdinal()
            .domain( labels )
            .range(["#8dd3c7", "#ffffb3", "#bebada", "#fb8072"])

        // Add X axis --> it is a date format
        var x = d3.scaleLinear()
            .domain([0, width_value])
            .range([0, width]);
        svg.append("g")
            .attr("transform", "translate(0," + height + ")")
            .call(d3.axisBottom(x));

        // Add Y axis
        var y = d3.scaleLinear()
            .domain([0, height_value])
            .range([height, 0]);
        svg.append("g")
            .call(d3.axisLeft(y));

        // Add the lines
        var line = d3.line()
            .x(function (d) { return x(+d.subject_label) })
            .y(function (d) { return y(+d.value) })
        svg.selectAll("myLines")
            .data(dataReady)
            .enter()
            .append("path")
            .attr("d", function (d) { return line(d.values) })
            // .attr("stroke", function (d) { return myColor(d.name) })
            .attr("stroke", function (d) { return labelColor(d.name) })
            .style("stroke-width", 1)
            .style("fill", "none")

        // Handmade legend
        svg.append("circle").attr("cx", 700).attr("cy", 30).attr("r", 6).style("fill", "#8dd3c7")
        svg.append("circle").attr("cx", 700).attr("cy", 60).attr("r", 6).style("fill", "#ffffb3")
        svg.append("circle").attr("cx", 700).attr("cy", 90).attr("r", 6).style("fill", "#bebada")
        svg.append("circle").attr("cx", 700).attr("cy", 120).attr("r", 6).style("fill", "#fb8072")

        svg.append("text").attr("x", 720).attr("y", 30).text("Calcium").style("font-size", "15px").attr("alignment-baseline", "middle")
        svg.append("text").attr("x", 720).attr("y", 60).text("Fibrous").style("font-size", "15px").attr("alignment-baseline", "middle")
        svg.append("text").attr("x", 720).attr("y", 90).text("IPH_lipid").style("font-size", "15px").attr("alignment-baseline", "middle")
        svg.append("text").attr("x", 720).attr("y", 120).text("IPH").style("font-size", "15px").attr("alignment-baseline", "middle")

        // Add the x Axis
        svg.append("g")
            .attr("transform", "translate(0," + height + ")")
            .call(d3.axisBottom(x));

        // text label for the x axis
        svg.append("text")
            .attr("transform",
                "translate(" + (width / 2) + " ," +
                (height + margin.top + 20) + ")")
            .style("text-anchor", "middle")
            .text("feature IDs");

        // Add the y Axis
        svg.append("g")
            .call(d3.axisLeft(y));

        // text label for the y axis
        svg.append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", 0 - margin.left)
            .attr("x", 0 - (height / 2))
            .attr("dy", "1em")
            .style("text-anchor", "middle")
            .text("featue Value");  


        // Add a legend at the end of each line
        // svg
        //     .selectAll("myLabels")
        //     .data(dataReady)
        //     .enter()
        //     .append('g')
        //     .append("text")
        //     .datum(function (d) { return { name: d.name, value: d.values[d.values.length - 1] }; }) // keep only the last value of each time series
        //     .attr("transform", function (d) { return "translate(" + x(d.value.subject_label) + "," + y(d.value.value) + ")"; }) // Put the text at the position of the last point
        //     .attr("x", 12) // shift the text a bit more right
        //     .text(function (d) { return d.name; })
        //     .style("fill", function (d) { return myColor(d.name) })
        //     .style("font-size", 10)

})
