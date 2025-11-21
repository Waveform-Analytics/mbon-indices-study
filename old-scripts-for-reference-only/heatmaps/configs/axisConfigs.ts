/**
 * Shared axis configurations for heatmap components
 */

export interface YAxisConfig {
  // Domain for the y-scale
  domain: [number, number];
  // Tick values to display on the axis
  tickValues: number[];
  // Function to format tick labels
  tickFormat: (value: number, index: number) => string;
  // Function to transform actual hour (0-23) to scale position
  transformHour: (hour: number) => number;
  // Height of each cell (based on number of rows)
  cellRows: number;
}

/**
 * Standard y-axis configuration (0:00 at top, 22:00 at bottom)
 */
export const STANDARD_Y_AXIS: YAxisConfig = {
  domain: [0, 22],
  tickValues: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
  tickFormat: (_: number, i: number) => {
    const hours = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22];
    return `${hours[i]}:00`;
  },
  transformHour: (hour: number) => hour,
  cellRows: 12, // 12 rows for 2-hour intervals
};

/**
 * Midnight-centered y-axis configuration
 * Places midnight (0:00) in the middle of the visualization
 * for better visualization of nighttime fish calling patterns
 *
 * Visual layout (top to bottom):
 * - 12:00 (noon)
 * - 14:00, 16:00, 18:00, 20:00, 22:00
 * - 0:00 (midnight) <- CENTER
 * - 2:00, 4:00, 6:00, 8:00, 10:00
 * - 12:00 (noon next day)
 */
export const MIDNIGHT_CENTERED_Y_AXIS: YAxisConfig = {
  // Domain spans from noon (-12) to 10:00 next day (10)
  domain: [-12, 10],
  // Tick positions for 2-hour intervals
  tickValues: [-11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9],
  tickFormat: (_: number, i: number) => {
    // Display hours in correct order: 12, 14, 16, 18, 20, 22, 0, 2, 4, 6, 8, 10
    const hours = [12, 14, 16, 18, 20, 22, 0, 2, 4, 6, 8, 10];
    return `${hours[i]}:00`;
  },
  transformHour: (hour: number) => {
    // Transform actual hour (0-23) to shifted position
    // Hours 0-11 stay as is, hours 12-23 shift to negative
    if (hour < 12) {
      return hour;
    } else {
      return hour - 24;
    }
  },
  cellRows: 12,
};

/**
 * Helper function to get inverse transform for tooltips
 * Converts from transformed hour back to original hour
 */
export const getOriginalHour = (transformedHour: number, config: YAxisConfig): number => {
  if (config === MIDNIGHT_CENTERED_Y_AXIS) {
    return transformedHour < 0 ? transformedHour + 24 : transformedHour;
  }
  return transformedHour;
};