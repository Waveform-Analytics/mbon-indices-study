'use client';

import React, { useMemo } from 'react';
import * as d3 from 'd3';
import BaseHeatmap, { ProcessedDataPoint } from './BaseHeatmap';
import { useHeatmapData } from './hooks/useHeatmapData';
import { MIDNIGHT_CENTERED_Y_AXIS } from './configs/axisConfigs';

interface EnvironmentalHeatmapProps {
  className?: string;
  useMidnightCentering?: boolean;
}

// Environmental variable configuration
const ENVIRONMENTAL_VARIABLES = [
  { key: 'Water temp (°C)', label: 'Temperature', unit: '°C', colorScheme: 'rdylbu' as const, reverseScale: true },
  { key: 'Water depth (m)', label: 'Depth', unit: 'm', colorScheme: 'blues' as const, reverseScale: false }
];

const EnvironmentalHeatmap: React.FC<EnvironmentalHeatmapProps> = ({
  className = '',
  useMidnightCentering = true
}) => {
  const {
    loading,
    error,
    stations,
    selectedStation,
    selectedVariable,
    setSelectedStation,
    setSelectedVariable,
    filteredData
  } = useHeatmapData({
    dataUrl: '/views/02_environmental_aligned_2021.json',
    excludeKeys: ['datetime', 'station', 'year'],
    defaultStation: '9M',
    defaultVariable: 'Water temp (°C)'
  });

  // Get current variable configuration
  const currentVarConfig = ENVIRONMENTAL_VARIABLES.find(v => v.key === selectedVariable) ||
                          ENVIRONMENTAL_VARIABLES[0];

  // Process data for heatmap
  const processedData = useMemo<ProcessedDataPoint[]>(() => {
    if (!filteredData.length || !selectedVariable) return [];

    const processed: ProcessedDataPoint[] = [];

    filteredData.forEach(d => {
      const value = d[selectedVariable] as number;
      // Only add data points with valid values
      if (value !== null && value !== undefined && !isNaN(value)) {
        const date = new Date(d.datetime);
        const dayOfYear = Math.floor(
          (date.getTime() - new Date(Date.UTC(date.getUTCFullYear(), 0, 0)).getTime()) / 86400000
        );

        processed.push({
          day: dayOfYear,
          hour: date.getUTCHours(),
          value: value,
          date: date
        });
      }
    });

    return processed;
  }, [filteredData, selectedVariable]);

  // Custom tooltip formatter for environmental data
  const formatTooltip = (d: ProcessedDataPoint) => {
    return `Date: ${d.date.toLocaleDateString()}<br/>
            Hour: ${d.originalHour || d.hour}:00<br/>
            ${currentVarConfig.label}: ${d.value.toFixed(2)} ${currentVarConfig.unit}`;
  };

  return (
    <div className={className}>
      {/* Selection controls */}
      <div className="flex gap-4 mb-4">
        <div>
          <label className="block text-sm font-medium mb-1">Station</label>
          <select
            className="px-3 py-2 border border-gray-200 rounded-md text-sm bg-background"
            value={selectedStation}
            onChange={(e) => setSelectedStation(e.target.value)}
          >
            {stations.map(station => (
              <option key={station} value={station}>{station}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Variable</label>
          <select
            className="px-3 py-2 border border-gray-200 rounded-md text-sm bg-background"
            value={selectedVariable}
            onChange={(e) => setSelectedVariable(e.target.value)}
          >
            {ENVIRONMENTAL_VARIABLES.map(variable => (
              <option key={variable.key} value={variable.key}>{variable.label}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Base heatmap */}
      <BaseHeatmap
        data={processedData}
        loading={loading}
        error={error}
        colorScheme={currentVarConfig.colorScheme}
        reverseColorScale={currentVarConfig.reverseScale}
        formatTooltip={formatTooltip}
        legendLabel={`${currentVarConfig.label} (${currentVarConfig.unit})`}
        legendFormatter={(v) => d3.format('.1f')(v)}
        yAxisConfig={useMidnightCentering ? MIDNIGHT_CENTERED_Y_AXIS : undefined}
        noDataMessage={`No ${currentVarConfig.label} data available for station ${selectedStation}`}
      />
    </div>
  );
};

export default EnvironmentalHeatmap;