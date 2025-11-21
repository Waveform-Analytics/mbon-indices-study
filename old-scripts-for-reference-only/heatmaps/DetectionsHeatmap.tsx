'use client';

import React, { useMemo } from 'react';
import * as d3 from 'd3';
import BaseHeatmap, { ProcessedDataPoint } from './BaseHeatmap';
import { useHeatmapData } from './hooks/useHeatmapData';
import { MIDNIGHT_CENTERED_Y_AXIS } from './configs/axisConfigs';

interface DetectionsHeatmapProps {
  className?: string;
  useMidnightCentering?: boolean;
}

const DetectionsHeatmap: React.FC<DetectionsHeatmapProps> = ({
  className = '',
  useMidnightCentering = true // Default to midnight centering for fish data
}) => {
  const {
    loading,
    error,
    stations,
    variables: species,
    selectedStation,
    selectedVariable: selectedSpecies,
    setSelectedStation,
    setSelectedVariable: setSelectedSpecies,
    filteredData
  } = useHeatmapData({
    dataUrl: '/views/02_detections_aligned_2021.json',
    excludeKeys: ['datetime', 'station', 'year'],
    defaultStation: '9M',
    defaultVariable: 'Silver perch'
  });

  // Process data for heatmap
  const processedData = useMemo<ProcessedDataPoint[]>(() => {
    if (!filteredData.length || !selectedSpecies || !species.includes(selectedSpecies)) {
      return [];
    }

    const processed = filteredData.map(d => {
      const date = new Date(d.datetime);
      const dayOfYear = Math.floor(
        (date.getTime() - new Date(Date.UTC(date.getUTCFullYear(), 0, 0)).getTime()) / 86400000
      );

      return {
        day: dayOfYear,
        hour: date.getUTCHours(),
        value: d[selectedSpecies] as number,
        date: date
      };
    }).filter(d => d.value !== null && d.value !== undefined && !isNaN(d.value));

    return processed;
  }, [filteredData, selectedSpecies, species]);

  // Custom tooltip formatter for fish detections
  const formatTooltip = (d: ProcessedDataPoint) => {
    return `Date: ${d.date.toLocaleDateString()}<br/>
            Hour: ${d.originalHour || d.hour}:00<br/>
            ${selectedSpecies}: ${d.value}`;
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
          <label className="block text-sm font-medium mb-1">Species</label>
          <select
            className="px-3 py-2 border border-gray-200 rounded-md text-sm bg-background"
            value={selectedSpecies}
            onChange={(e) => setSelectedSpecies(e.target.value)}
          >
            {species.map(sp => (
              <option key={sp} value={sp}>{sp}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Base heatmap */}
      <BaseHeatmap
        data={processedData}
        loading={loading}
        error={error}
        colorScheme="viridis"
        formatTooltip={formatTooltip}
        legendLabel={selectedSpecies}
        legendFormatter={(v) => d3.format('.0f')(v)}
        yAxisConfig={useMidnightCentering ? MIDNIGHT_CENTERED_Y_AXIS : undefined}
        noDataMessage={`No ${selectedSpecies} data available for station ${selectedStation}`}
      />
    </div>
  );
};

export default DetectionsHeatmap;