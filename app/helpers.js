"use strict";

// functions for drawing
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
        
function print_time(dates) {
    networks.selectAll("text.date").remove();
    var x = networks_left + 2 * pad,
        y = networks_top + pad,
        lines = [date_str(dates[0], dates[1]) + " - ", date_str(dates[2], dates[3])];
    
    print_text_lines(networks, x, y, date_size, pad, lines, "date", "left");
    y += date_size;
}

function date_str(month, year) {
    return parseInt(year) + " " + month_names[month];
}

function year_to_session(year) {
    return Math.floor((year - 1787) / 2);
}


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


// Functions to initialize and plot all of the display
//----------------------------------------------------------------//
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

function init_network() {
    networks.selectAll("circle.node").data(d3.values(nodes[parseInt(current_time)]), function(d,i) {return d[node_keys['id']];})
        .enter().append("circle")
        .attr("cx", function(d,i) {/*console.log(networks_x(d[node_keys['x']]));*/ return networks_x(d[node_keys['x']]);})
        .attr("cy", function(d,i) {/*console.log(networks_y(d[node_keys['y']]));*/ return networks_y(d[node_keys['y']]);})
        .attr("r", function(d,i) {/*console.log(age_scale(d[node_keys['age']]));*/ return node_size(d[node_keys['age']]);})
        .attr("class", node_class);
    
}

function plot_network(duration=1000) {
    var node_plot = networks.selectAll("circle.node").data(d3.values(nodes[parseInt(current_time)]), 
                                                           function(d,i) {return d[node_keys['id']];})
    node_plot.enter().append("circle")
        .attr("cx", function(d,i) {/*console.log(networks_x(d[node_keys['x']]));*/ return networks_x(d[node_keys['x']]);})
        .attr("cy", function(d,i) {/*console.log(networks_y(d[node_keys['y']]));*/ return networks_y(d[node_keys['y']]);})
        .attr("r", function(d,i) {/*console.log(age_scale(d[node_keys['age']]));*/ return node_size(d[node_keys['age']]);})
        .attr("class", node_class);
    node_plot.exit().transition().duration(duration * 0.5).remove();
    node_plot.transition().duration(duration)
//        .each("start", function() {  // Start animation
//                            d3.select(this)  // 'this' means the current element
//                                
//                        })
        .delay(function(d, i) { return 15*d[node_keys['experience']];})
//                            return i / dataset.length * 500;  // Dynamic delay (i.e. each item delays a little longer)
//                        })
        .attr("cx", function(d,i) {/*console.log(networks_x(d[node_keys['x']]));*/ return networks_x(d[node_keys['x']]);})
        .attr("cy", function(d,i) {/*console.log(networks_y(d[node_keys['y']]));*/ return networks_y(d[node_keys['y']]);})
        .attr("r", function(d,i) {/*console.log(age_scale(d[node_keys['age']]));*/ return node_size(d[node_keys['age']]);})
        .attr("class",function(d,i) {return 'node ' + party_class[senators[d[node_keys['id']]]['party']]});
    
   
}
