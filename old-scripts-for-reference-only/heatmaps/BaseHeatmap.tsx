'use client';

import React, { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import * as d3 from 'd3';
import { YAxisConfig, STANDARD_Y_AXIS } from './configs/axisConfigs';

export interface ProcessedDataPoint {
  day: number;
  hour: number;
  value: number;
  date: Date;
  originalHour?: number; // Store original hour for tooltips
}

export interface BaseHeatmapProps {
  // Processed data ready for visualization
  data: ProcessedDataPoint[];
  // Optional dimensions
  width?: number;
  height?: number;
  // Optional margins
  margins?: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  };
  // Color scale configuration
  colorScale?: (value: number) => string;
  colorDomain?: [number, number];
  colorScheme?: 'viridis' | 'blues' | 'rdylbu' | 'plasma' | 'turbo';
  reverseColorScale?: boolean;
  // Tooltip formatter
  formatTooltip?: (d: ProcessedDataPoint) => string;
  // Legend configuration
  legendLabel?: string;
  legendFormatter?: (value: number) => string;
  // Categorical legend support
  categoricalLegend?: boolean;
  categoryLabels?: string[];
  categoryColors?: string[];
  // Y-axis configuration (for midnight centering)
  yAxisConfig?: YAxisConfig;
  // Additional styling
  className?: string;
  // Loading/error states
  loading?: boolean;
  error?: string | null;
  // No data message
  noDataMessage?: string;
}

const BaseHeatmap: React.FC<BaseHeatmapProps> = React.memo(({
  data,
  width: customWidth,
  height: customHeight = 450,
  margins = { top: 80, right: 50, bottom: 140, left: 70 },
  colorScale: customColorScale,
  colorDomain: customColorDomain,
  colorScheme = 'viridis',
  reverseColorScale = false,
  formatTooltip,
  legendLabel = 'Value',
  legendFormatter = (v: number) => d3.format('.3f')(v),
  categoricalLegend = false,
  categoryLabels = [],
  categoryColors = [],
  yAxisConfig = STANDARD_Y_AXIS,
  className = '',
  loading = false,
  error = null,
  noDataMessage = 'No data available',
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState(800); // Start with default width

  // Handle container resize with better initial detection
  useEffect(() => {
    const updateWidth = () => {
      if (containerRef.current) {
        const newWidth = containerRef.current.clientWidth;
        if (newWidth > 0 && newWidth !== containerWidth) {
          setContainerWidth(newWidth);
        }
      }
    };

    // Immediate width detection
    updateWidth();

    // Try multiple times to detect width, especially for components lower on page
    const timers = [
      setTimeout(updateWidth, 10),
      setTimeout(updateWidth, 50),
      setTimeout(updateWidth, 100),
      setTimeout(updateWidth, 200),
      setTimeout(updateWidth, 500),
      setTimeout(updateWidth, 1000)
    ];

    // Use ResizeObserver for more reliable detection
    let resizeObserver: ResizeObserver | null = null;
    if (containerRef.current && 'ResizeObserver' in window) {
      resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const newWidth = entry.contentRect.width;
          if (newWidth > 0 && newWidth !== containerWidth) {
            setContainerWidth(newWidth);
          }
        }
      });
      resizeObserver.observe(containerRef.current);
    }

    // Also listen for window resize events
    window.addEventListener('resize', updateWidth);

    // Additional check when page finishes loading
    const handleLoad = () => {
      setTimeout(updateWidth, 100);
    };
    window.addEventListener('load', handleLoad);

    return () => {
      timers.forEach(clearTimeout);
      window.removeEventListener('resize', updateWidth);
      window.removeEventListener('load', handleLoad);
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
    };
  }, [containerWidth]); // Include containerWidth for comparison

  // Process data with y-axis transformation
  const processedData = React.useMemo(() => {
    return data.map(d => ({
      ...d,
      originalHour: d.hour, // Store original for tooltips
      hour: yAxisConfig.transformHour(d.hour), // Transform for display
    }));
  }, [data, yAxisConfig]);

  // Create color scale - memoize the domain calculation separately to avoid re-creating scale
  const colorDomain = useMemo(() => {
    if (customColorDomain) return customColorDomain;
    if (!processedData.length) return [0, 1];
    return d3.extent(processedData, d => d.value) as [number, number] || [0, 1];
  }, [customColorDomain, processedData]);

  const getColorScale = useCallback(() => {
    if (customColorScale) {
      return customColorScale;
    }

    const finalDomain = reverseColorScale ? [colorDomain[1], colorDomain[0]] : colorDomain;

    const interpolators: Record<string, (t: number) => string> = {
      viridis: d3.interpolateViridis,
      blues: d3.interpolateBlues,
      rdylbu: d3.interpolateRdYlBu,
      plasma: d3.interpolatePlasma,
      turbo: d3.interpolateTurbo,
    };

    return d3.scaleSequential(interpolators[colorScheme] || d3.interpolateViridis)
      .domain(finalDomain);
  }, [customColorScale, colorDomain, reverseColorScale, colorScheme]);

  // Render the heatmap
  useEffect(() => {
    if (!processedData.length || !svgRef.current || containerWidth < 100) return;

    // Defensive check for valid data
    const validData = processedData.filter(d =>
      d.value !== null &&
      d.value !== undefined &&
      !isNaN(d.value) &&
      d.day !== null &&
      d.day !== undefined &&
      d.hour !== null &&
      d.hour !== undefined
    );

    if (!validData.length) return;

    // Clear previous chart
    d3.select(svgRef.current).selectAll('*').remove();

    const width = (customWidth || containerWidth) - margins.left - margins.right;
    const height = customHeight - margins.top - margins.bottom;

    // Create SVG
    const svg = d3.select(svgRef.current)
      .attr('width', customWidth || containerWidth)
      .attr('height', customHeight);

    const g = svg.append('g')
      .attr('transform', `translate(${margins.left},${margins.top})`);

    // Create scales
    const dayExtent = d3.extent(validData, d => d.day) as [number, number];
    const xScale = d3.scaleLinear()
      .domain(dayExtent)
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain(yAxisConfig.domain)
      .range([height, 0]);

    const colorScale = getColorScale();

    // Calculate cell dimensions
    const cellWidth = width / (dayExtent[1] - dayExtent[0] + 1);
    const cellHeight = height / yAxisConfig.cellRows;

    // Add cells
    g.selectAll('rect')
      .data(validData)
      .enter()
      .append('rect')
      .attr('x', d => xScale(d.day))
      .attr('y', d => yScale(d.hour) - cellHeight)
      .attr('width', cellWidth)
      .attr('height', cellHeight)
      .attr('fill', d => {
        if (typeof colorScale === 'function') {
          return colorScale(d.value);
        }
        return '#000';
      })
      .attr('stroke', 'none')
      .on('mouseover', function(event, d) {
        // Add tooltip
        const tooltip = d3.select('body').append('div')
          .attr('class', 'heatmap-tooltip')
          .style('position', 'absolute')
          .style('padding', '10px')
          .style('background', 'rgba(0, 0, 0, 0.8)')
          .style('color', 'white')
          .style('border-radius', '5px')
          .style('pointer-events', 'none')
          .style('opacity', 0)
          .style('z-index', 9999);

        tooltip.transition()
          .duration(200)
          .style('opacity', 0.9);

        const tooltipContent = formatTooltip
          ? formatTooltip(d)
          : `Date: ${d.date.toLocaleDateString()}<br/>
             Hour: ${d.originalHour || d.hour}:00<br/>
             ${legendLabel}: ${d.value.toFixed(3)}`;

        tooltip.html(tooltipContent)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 28) + 'px');
      })
      .on('mouseout', function() {
        d3.selectAll('.heatmap-tooltip').remove();
      });

    // Add X axis
    const xAxis = d3.axisBottom(xScale)
      .tickFormat(d => {
        const year = validData[0]?.date.getUTCFullYear() || new Date().getFullYear();
        const date = new Date(Date.UTC(year, 0, Number(d)));
        return d3.timeFormat('%b %d')(date);
      })
      .ticks(10);

    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(xAxis)
      .selectAll('text')
      .style('text-anchor', 'end')
      .attr('dx', '-.8em')
      .attr('dy', '.8em')
      .attr('transform', 'rotate(-45)');

    // Add Y axis using config
    const yAxis = d3.axisLeft(yScale)
      .tickValues(yAxisConfig.tickValues)
      .tickFormat((d, i) => yAxisConfig.tickFormat(Number(d), i))
      .tickSize(0);

    g.append('g')
      .call(yAxis);

    // Add X axis label
    svg.append('text')
      .attr('transform', `translate(${(customWidth || containerWidth) / 2},${margins.top + height + 90})`)
      .style('text-anchor', 'middle')
      .text('Date');

    // Add Y axis label
    svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', 15)
      .attr('x', -(margins.top + height / 2))
      .style('text-anchor', 'middle')
      .text('Hour of Day');

    // Add color legend
    const legendWidth = categoricalLegend ? 400 : 200; // Wider for categorical legends
    const legendHeight = 20;
    
    // Position legend based on type
    const legendX = categoricalLegend 
      ? Math.max(20, (customWidth || containerWidth) / 2 - legendWidth / 2) // Center for categorical
      : (customWidth || containerWidth) - legendWidth - 20; // Right-aligned for continuous
    
    const legend = svg.append('g')
      .attr('transform', `translate(${legendX},${15})`);

    if (categoricalLegend && categoryLabels.length > 0 && categoryColors.length > 0) {
      // Traditional horizontal legend (like typical chart legends)
      const squareSize = 12;
      const itemSpacing = 20; // Space between legend items
      const textOffset = 5; // Space between square and text
      
      let currentX = 0;
      
      categoryLabels.forEach((label, i) => {
        const color = categoryColors[i] || '#000000';
        
        // Color square
        legend.append('rect')
          .attr('x', currentX)
          .attr('y', 4) // Center vertically with text
          .attr('width', squareSize)
          .attr('height', squareSize)
          .style('fill', color)
          .style('stroke', '#333')
          .style('stroke-width', 0.5);
          
        // Label text
        const text = legend.append('text')
          .attr('x', currentX + squareSize + textOffset)
          .attr('y', 10) // Center vertically
          .attr('dy', '0.35em')
          .style('font-size', '12px')
          .style('font-weight', '500')
          .style('fill', '#333')
          .text(label);
          
        // Calculate width of this text for next item positioning
        const textWidth = (text.node() as SVGTextElement)?.getBBox().width || 80;
        currentX += squareSize + textOffset + textWidth + itemSpacing;
      });
    } else {
      // Continuous legend (existing logic)
      const colorScaleDomain = 'domain' in colorScale && typeof colorScale.domain === 'function'
        ? colorScale.domain()
        : (customColorDomain || (d3.extent(validData, d => d.value) as [number, number]) || [0, 1]);
      const legendDomain = reverseColorScale
        ? [colorScaleDomain[1], colorScaleDomain[0]]
        : colorScaleDomain;

      const legendScale = d3.scaleLinear()
        .domain(legendDomain)
        .range([0, legendWidth]);

      const legendAxis = d3.axisBottom(legendScale)
        .ticks(5)
        .tickFormat(d => legendFormatter(Number(d)));

      // Create gradient for legend
      const gradientId = 'gradient-' + Math.random();
      const gradient = svg.append('defs')
        .append('linearGradient')
        .attr('id', gradientId)
        .attr('x1', '0%')
        .attr('x2', '100%')
        .attr('y1', '0%')
        .attr('y2', '0%');

      const nStops = 10;
      const colorRange = d3.range(nStops).map(i => i / (nStops - 1));

      colorRange.forEach(t => {
        gradient.append('stop')
          .attr('offset', `${t * 100}%`)
          .attr('stop-color', colorScale(legendDomain[0] + t * (legendDomain[1] - legendDomain[0])));
      });

      legend.append('rect')
        .attr('width', legendWidth)
        .attr('height', legendHeight)
        .style('fill', `url(#${gradientId})`);

      legend.append('g')
        .attr('transform', `translate(0,${legendHeight})`)
        .call(legendAxis);
    }
    
    // Add legend title
    legend.append('text')
      .attr('x', legendWidth / 2)
      .attr('y', -5)
      .style('text-anchor', 'middle')
      .style('font-size', '12px')
      .text(legendLabel);

  }, [processedData, containerWidth, customWidth, customHeight, margins, getColorScale,
      formatTooltip, legendLabel, legendFormatter, yAxisConfig, reverseColorScale, customColorDomain,
      categoricalLegend, categoryLabels, categoryColors]);

  // Handle loading state
  if (loading) {
    return (
      <div className={`flex items-center justify-center h-64 ${className}`}>
        <div className="text-muted-foreground">Loading data...</div>
      </div>
    );
  }

  // Handle error state
  if (error) {
    return (
      <div className={`flex items-center justify-center h-64 ${className}`}>
        <div className="text-red-500">Error: {error}</div>
      </div>
    );
  }

  // Handle no data
  if (!data.length) {
    return (
      <div className={`flex items-center justify-center h-64 ${className}`}>
        <div className="text-muted-foreground">{noDataMessage}</div>
      </div>
    );
  }

  return (
    <div ref={containerRef} className={`w-full ${className}`}>
      <svg ref={svgRef}></svg>
    </div>
  );
});

BaseHeatmap.displayName = 'BaseHeatmap';

export default BaseHeatmap;