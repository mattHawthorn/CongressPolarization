"use strict";

// Formatting  functions
//---------------------------------------------------------------//

function month_and_year(time) {
    var year = Math.floor(time/12.0);
    var month = time % 12;
    if (month === 0) {
        month = 12;
        year = year - 1;
    }
    return [month, year];
}


function get_date_range(x) {
    var time = timelines_x.invert(x),
        year1 = Math.floor(time),
        month1 = Math.floor((time - year1) * 12.0) + 1;
    time = timelines_x.invert(x) + time_lag;
    
    var year2 = Math.floor(time),
        month2 = Math.floor((time - year2) * 12.0);
    if (month2 === 0) {
        month2 = 12;
        year2 = year2 - 1;
    }
    return [month1, year1, month2, year2];
}


function date_str(month, year) {
    return parseInt(year) + " " + month_names[month];
}

function year_to_session(year) {
    return Math.floor((year - 1787) / 2);
}


function print_info(month, year) {
    if (year % 2 === 1) {
        if (month > 0.5 * time_lag) {
            var session = year_to_session(year);
        } else {
            var session = year_to_session(year) - 1;
        } 
    } else {
        var session = year_to_session(year);
    }
    
    var last_digit = session % 10;
    var suffix = numeric_suffixes[last_digit];
    var session_str = parseInt(session) + suffix;
    var session_title = session_str + " Congress";
    var wiki_link = "https://en.wikipedia.org/wiki/" + session_str + "_United_States_Congress#Major_events";
    
    info.selectAll("text").remove();
    
    y = info_y(0) + info_title_size;
    
    info.select("a").remove();
    
    var title_link = info.append("a").attr("xlink:href", wiki_link).attr("target", "_blank");
    title_link.append("text").attr("x", info_x(0)).attr("y", y)
        .attr("text-anchor", "left").attr("class", "info_title")
        .style("font-size", font_size(info_title_size))
        .text(session_title);
    
    //print_text_lines(networks, x, y, date_size, pad, lines, "date", "left");
    print_text_lines(info, info_x(info_indent), y + 2*info_sep, info_text_size, info_sep, ['Narration','goes','here.'], "info_text","left")
}
    

function font_size(size) {
    return parseInt(size) + "px";
}


// print lines of text sequentially from a start location
function print_text_lines(elt, x, y, size, sep, lines, cl, anchor='left',clear=true) {
    if (sep === undefined) {
        sep = 0.2 * size;
    }

    if (clear) {
        var t = elt.selectAll("text."+cl);
        t.remove();
    }
    
    y += size;
    var t = elt.selectAll("text."+cl).data(lines);
    t.enter().append("text")
        .attr("x",x)
        .attr("y", function(d,i){return y + i*(size + sep)})
        .style("font-size", size)
        .attr("class", cl)
        .style("text-anchor", anchor)
        .text(function(d,i){return d});
}




// Control and plotting
//----------------------------------------------------------------//

function increment_time(increment) {
    var time = current_time + increment;
    
    if(time >= min_time && time <= max_time){
        var date = month_and_year(time);
        current_time = time;
        current_month = date[0];
        current_year = date[1];
        timeWindow.attr("x", timelines_x(current_time - time_lag));
        return true;
    }
    return false;
}


function set_network_vars() {
    current_nodes = nodes[parseInt(current_time)];
    current_edges = edges[parseInt(current_time)].filter(function(edge) { return edge[2] >= edge_threshold});    
}

  
function print_time(dates) {
    networks.selectAll("text.date").remove();
    var x = networks_left + 2 * pad,
        y = networks_top + pad,
        lines = [date_str(dates[0], dates[1]) + " - ", date_str(dates[2], dates[3])];
    
    print_text_lines(networks, x, y, date_size, pad, lines, "date", "left");
    y += date_size;
}


function update_text() {
    var dates = month_and_year(current_time - time_lag + 1).concat(month_and_year(current_time));
    print_time(dates);
    print_info(current_month,current_year);
}

function node_class(d,i) {
    if (current_time > min_time) {
        var enter_status = (d[node_keys['id']] in nodes[parseInt(current_time - 1)] ? '' : 'enter_node ');
    } else {
        var enter_status = '';
    }
    if (current_time < max_time) {
        var exit_status = (d[node_keys['id']] in nodes[parseInt(current_time + 1)] ? '' : 'exit_node ');
    } else {
        var exit_status = '';
    }
    return 'node ' + enter_status + exit_status + party_class[senators[d[node_keys['id']]]['party']];
}


function edge_class(d,i) {
    return 'edge ' + party_class[senators[d[0]]['party'] + ' ' + senators[d[1]]['party']];
}



function reset_window(x_mi,x_ma,y_mi,y_ma) {
console.log(x_mi,x_ma);
    if (x_mi === undefined) {
        x_min = ref_x_min;
    } else {x_min = x_mi;};
    if (x_ma === undefined) {
        x_max = ref_x_max;
    } else {x_max = x_ma;};
    if (y_mi === undefined) {
        y_min = ref_y_min;
    } else {y_min = y_mi;};
    if (y_ma === undefined) {
        y_max = ref_y_max;
    } else {y_max = y_ma;};
    console.log(x_min,x_max,y_min,y_max);
    networks_x = d3.scale.linear().domain([x_min,x_max])
            .range([networks_left + 0.5*(c - plot_w) + networks_offset, networks_left + c - 0.5*(c - plot_w) + networks_offset]);
    networks_y = d3.scale.linear().domain([y_min,y_max])
                                .range([networks_top + networks_h - pad, networks_top + pad]);
    plot_network();
}


function init_network() {
    networks.selectAll("line.edge").data(current_edges,
                                        function(d) {return d[0] + ' ' + d[1];})
        .enter().append("line")//.filter(function(d) { return d[2] > edge_threshold;})
        .attr("x1", function(d,i) {return networks_x(current_nodes[d[0]][node_keys['x']]);})
        .attr("y1", function(d,i) {return networks_y(current_nodes[d[0]][node_keys['y']]);})
        .attr("x2", function(d,i) {return networks_x(current_nodes[d[1]][node_keys['x']]);})
        .attr("y2", function(d,i) {return networks_y(current_nodes[d[1]][node_keys['y']]);})
        .style("stroke-width", function(d,i) { return edge_width(d[2]);})
        .attr("class", edge_class);
        
    
    networks.selectAll("circle.node").data(d3.values(current_nodes), function(d,i) {return d[node_keys['id']];})
        .enter().append("circle")
        .attr("cx", function(d,i) {/*console.log(networks_x(d[node_keys['x']]));*/ return networks_x(d[node_keys['x']]);})
        .attr("cy", function(d,i) {/*console.log(networks_y(d[node_keys['y']]));*/ return networks_y(d[node_keys['y']]);})
        .attr("r", function(d,i) {/*console.log(age_scale(d[node_keys['age']]));*/ return node_size(d[node_keys['age']]);})
        .attr("class", node_class);
    
}


function plot_network(duration=500) {
    //networks.selectAll("line.edge").remove();
    
    var node_plot = networks.selectAll("circle.node").data(d3.values(current_nodes), 
                                                           function(d) {return d[node_keys['id']];})
    node_plot.enter().append("circle")
        .attr("cx", function(d,i) {/*console.log(networks_x(d[node_keys['x']]));*/ return networks_x(d[node_keys['x']]);})
        .attr("cy", function(d,i) {/*console.log(networks_y(d[node_keys['y']]));*/ return networks_y(d[node_keys['y']]);})
        .attr("r", 0.0)
        .attr("class", node_class)
        .on("mouseover", function(d){
          var senator = senators[d[0]];
          var name = senator["first_name"] + " " + senator["last_name"];
          var age = d[1];
          var experience = d[2];
          var party = senator["party"]
          var state = senator["state"]
          var lines = [name, party + ", " + state, "Age: " + age.toString(), "Yrs. exp.: " + Math.round(experience).toString()];
          var coords = d3.mouse(this);
          console.log(name);
          print_text_lines(networks,coords[0],coords[1],20,5,lines,"info_text");
//          console.log(name);
//          networks.append("text")
//              .attr("x",coords[0])
//              .attr("y",coords[1])
//              .text(name).attr("class","info_text");
        })
        .on("mouseout", function(){
          networks.selectAll("text.info_text").remove();
        });
    node_plot.exit().transition().duration(duration * 0.3).remove();
    node_plot.transition().duration(duration * 0.7)
        //.delay(function(d, i) { return 15*d[node_keys['experience']];}) // most experienced move last
        .attr("cx", function(d,i) {/*console.log(networks_x(d[node_keys['x']]));*/ return networks_x(d[node_keys['x']]);})
        .attr("cy", function(d,i) {/*console.log(networks_y(d[node_keys['y']]));*/ return networks_y(d[node_keys['y']]);})
        .attr("r", function(d,i) {/*console.log(age_scale(d[node_keys['age']]));*/ return node_size(d[node_keys['age']]);})
        .attr("class",function(d,i) {return 'node ' + party_class[senators[d[node_keys['id']]]['party']]});
   
    var edge_plot = networks.selectAll("line.edge").data(current_edges,
                                                        function(d) {return d[0] + ' ' + d[1];});

    edge_plot.exit().transition().duration(duration * 0.3).remove();
    
    edge_plot.transition().duration(duration)
        .attr("x1", function(d,i) {return networks_x(current_nodes[d[0]][node_keys['x']]);})
        .attr("y1", function(d,i) {return networks_y(current_nodes[d[0]][node_keys['y']]);})
        .attr("x2", function(d,i) {return networks_x(current_nodes[d[1]][node_keys['x']]);})
        .attr("y2", function(d,i) {return networks_y(current_nodes[d[1]][node_keys['y']]);})
        .style("stroke-width", function(d,i) { return edge_width(d[2]);})
        .attr("class", edge_class);
        //.style("opacity",0.5);
    edge_plot.enter().append("line").transition().duration(duration)//.filter(function(d) { return d[2] > edge_threshold;})
        .attr("x1", function(d,i) {return networks_x(current_nodes[d[0]][node_keys['x']]);})
        .attr("y1", function(d,i) {return networks_y(current_nodes[d[0]][node_keys['y']]);})
        .attr("x2", function(d,i) {return networks_x(current_nodes[d[1]][node_keys['x']]);})
        .attr("y2", function(d,i) {return networks_y(current_nodes[d[1]][node_keys['y']]);})
        .style("stroke-width", function(d,i) { return edge_width(d[2]);})
        .attr("class", edge_class);
        //.style("opacity", 0.0);
        
    
}
