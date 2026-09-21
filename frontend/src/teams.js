// Static team metadata. IDs match backend/Data/NBA_TEAMS.csv, so the team
// pickers work before the API has woken up. Colors and conferences aren't in
// the API; `colors` lists the primary first, then alternates for clashes.
export const TEAMS = [
  { id: 1610612737, abbreviation: 'ATL', name: 'Atlanta Hawks', nickname: 'Hawks', conference: 'East', colors: ['#E03A3E', '#26282A', '#C1D32F'] },
  { id: 1610612738, abbreviation: 'BOS', name: 'Boston Celtics', nickname: 'Celtics', conference: 'East', colors: ['#007A33', '#BA9653', '#000000'] },
  { id: 1610612739, abbreviation: 'CLE', name: 'Cleveland Cavaliers', nickname: 'Cavaliers', conference: 'East', colors: ['#860038', '#FDBB30', '#041E42'] },
  { id: 1610612740, abbreviation: 'NOP', name: 'New Orleans Pelicans', nickname: 'Pelicans', conference: 'West', colors: ['#0C2340', '#C8102E', '#85714D'] },
  { id: 1610612741, abbreviation: 'CHI', name: 'Chicago Bulls', nickname: 'Bulls', conference: 'East', colors: ['#CE1141', '#000000'] },
  { id: 1610612742, abbreviation: 'DAL', name: 'Dallas Mavericks', nickname: 'Mavericks', conference: 'West', colors: ['#00538C', '#B8C4CA', '#002B5E'] },
  { id: 1610612743, abbreviation: 'DEN', name: 'Denver Nuggets', nickname: 'Nuggets', conference: 'West', colors: ['#0E2240', '#FEC524', '#8B2131'] },
  { id: 1610612744, abbreviation: 'GSW', name: 'Golden State Warriors', nickname: 'Warriors', conference: 'West', colors: ['#1D428A', '#FFC72C'] },
  { id: 1610612745, abbreviation: 'HOU', name: 'Houston Rockets', nickname: 'Rockets', conference: 'West', colors: ['#CE1141', '#000000', '#C4CED4'] },
  { id: 1610612746, abbreviation: 'LAC', name: 'Los Angeles Clippers', nickname: 'Clippers', conference: 'West', colors: ['#C8102E', '#1D428A', '#BEC0C2'] },
  { id: 1610612747, abbreviation: 'LAL', name: 'Los Angeles Lakers', nickname: 'Lakers', conference: 'West', colors: ['#552583', '#FDB927', '#000000'] },
  { id: 1610612748, abbreviation: 'MIA', name: 'Miami Heat', nickname: 'Heat', conference: 'East', colors: ['#98002E', '#F9A01B', '#000000'] },
  { id: 1610612749, abbreviation: 'MIL', name: 'Milwaukee Bucks', nickname: 'Bucks', conference: 'East', colors: ['#00471B', '#0077C0', '#000000'] },
  { id: 1610612750, abbreviation: 'MIN', name: 'Minnesota Timberwolves', nickname: 'Timberwolves', conference: 'West', colors: ['#0C2340', '#78BE20', '#236192'] },
  { id: 1610612751, abbreviation: 'BKN', name: 'Brooklyn Nets', nickname: 'Nets', conference: 'East', colors: ['#000000', '#777D84'] },
  { id: 1610612752, abbreviation: 'NYK', name: 'New York Knicks', nickname: 'Knicks', conference: 'East', colors: ['#006BB6', '#F58426', '#BEC0C2'] },
  { id: 1610612753, abbreviation: 'ORL', name: 'Orlando Magic', nickname: 'Magic', conference: 'East', colors: ['#0077C0', '#000000', '#C4CED4'] },
  { id: 1610612754, abbreviation: 'IND', name: 'Indiana Pacers', nickname: 'Pacers', conference: 'East', colors: ['#002D62', '#FDBB30', '#BEC0C2'] },
  { id: 1610612755, abbreviation: 'PHI', name: 'Philadelphia 76ers', nickname: '76ers', conference: 'East', colors: ['#006BB6', '#ED174C', '#002B5C'] },
  { id: 1610612756, abbreviation: 'PHX', name: 'Phoenix Suns', nickname: 'Suns', conference: 'West', colors: ['#1D1160', '#E56020', '#000000'] },
  { id: 1610612757, abbreviation: 'POR', name: 'Portland Trail Blazers', nickname: 'Trail Blazers', conference: 'West', colors: ['#E03A3E', '#000000'] },
  { id: 1610612758, abbreviation: 'SAC', name: 'Sacramento Kings', nickname: 'Kings', conference: 'West', colors: ['#5A2D81', '#63727A', '#000000'] },
  { id: 1610612759, abbreviation: 'SAS', name: 'San Antonio Spurs', nickname: 'Spurs', conference: 'West', colors: ['#C4CED4', '#000000'] },
  { id: 1610612760, abbreviation: 'OKC', name: 'Oklahoma City Thunder', nickname: 'Thunder', conference: 'West', colors: ['#007AC1', '#EF3B24', '#002D62'] },
  { id: 1610612761, abbreviation: 'TOR', name: 'Toronto Raptors', nickname: 'Raptors', conference: 'East', colors: ['#CE1141', '#000000', '#A1A1A4'] },
  { id: 1610612762, abbreviation: 'UTA', name: 'Utah Jazz', nickname: 'Jazz', conference: 'West', colors: ['#F9A01B', '#000000', '#002B5C'] },
  { id: 1610612763, abbreviation: 'MEM', name: 'Memphis Grizzlies', nickname: 'Grizzlies', conference: 'West', colors: ['#5D76A9', '#12173F', '#F5B112'] },
  { id: 1610612764, abbreviation: 'WAS', name: 'Washington Wizards', nickname: 'Wizards', conference: 'East', colors: ['#002B5C', '#E31837', '#C4CED4'] },
  { id: 1610612765, abbreviation: 'DET', name: 'Detroit Pistons', nickname: 'Pistons', conference: 'East', colors: ['#1D42BA', '#C8102E', '#BEC0C2'] },
  { id: 1610612766, abbreviation: 'CHA', name: 'Charlotte Hornets', nickname: 'Hornets', conference: 'East', colors: ['#00788C', '#1D1160', '#A1A1A4'] },
];

const TEAM_META = Object.fromEntries(TEAMS.map((team) => [team.id, team]));

// The API decides which teams exist and what they're called; we add colors.
export const withTeamMeta = (apiTeams) =>
  apiTeams.map((team) => ({
    nickname: team.name.split(' ').pop(),
    conference: null,
    colors: ['#6B7280', '#111827'],
    ...TEAM_META[team.id],
    id: team.id,
    abbreviation: team.abbreviation,
    name: team.name,
  }));

// --- Color helpers -----------------------------------------------------------

const toLinear = (channel) => {
  const c = channel / 255;
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
};

const linearRgb = (hex) => [1, 3, 5].map((i) => toLinear(parseInt(hex.slice(i, i + 2), 16)));

const oklab = (hex) => {
  const [r, g, b] = linearRgb(hex);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [
    0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s,
  ];
};

const distance = (a, b) => {
  const [L1, a1, b1] = oklab(a);
  const [L2, a2, b2] = oklab(b);
  return 100 * Math.hypot(L1 - L2, a1 - a2, b1 - b2);
};

// Two fills clash when they're close in OKLab, or are both saturated shades
// of the same hue (navy next to royal blue reads as "one team").
const clashes = (a, b) => {
  const [, a1, b1] = oklab(a);
  const [, a2, b2] = oklab(b);
  const bothSaturated = Math.hypot(a1, b1) > 0.06 && Math.hypot(a2, b2) > 0.06;
  const hueGap = Math.abs(Math.atan2(b1, a1) - Math.atan2(b2, a2)) * (180 / Math.PI);
  const sameHue = Math.min(hueGap, 360 - hueGap) < 35;
  return distance(a, b) < 20 || (bothSaturated && sameHue);
};

// Like a road jersey: when colors clash, the away team switches to an alternate.
export const matchupColors = (away, home) => {
  const homeColor = home.colors[0];
  if (!clashes(away.colors[0], homeColor)) return { away: away.colors[0], home: homeColor };

  const alternate =
    away.colors.slice(1).find((color) => !clashes(color, homeColor)) ??
    [...away.colors].sort((a, b) => distance(b, homeColor) - distance(a, homeColor))[0];
  return { away: alternate, home: homeColor };
};

// Court lines use white on dark fills and black on light ones.
export const inkFor = (hex) => {
  const [r, g, b] = linearRgb(hex);
  const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return 1.05 / (luminance + 0.05) >= 3 ? '#FFFFFF' : '#000000';
};
