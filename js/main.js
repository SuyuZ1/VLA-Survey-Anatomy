let allData = [];
let filteredData = [];
let latestAllData = [];
let currentPage = 1;
let recordsPerPage = 10;
let sortColumn = null;
let sortDirection = 'asc';

// Flexible date parsing function
function parseFlexibleDate(raw) {
    if (raw === null || raw === undefined) return null;

    // If already Date
    if (raw instanceof Date) return isNaN(raw.getTime()) ? null : raw;

    // If number (e.g., 2025)
    if (typeof raw === 'number' && Number.isFinite(raw)) {
        return new Date(raw, 0, 1);
    }

    // Normalize to string and trim BOM + whitespace
    let str = String(raw).replace(/^\uFEFF/, '').trim();
    if (!str) return null;

    // Replace various dash-like Unicode chars with ASCII hyphen, and normalize other separators to '-'
    // covers: –, —, −, —, fullwidth '－', slashes, dots, spaces
    str = str.replace(/[\u2012-\u2015\u2212\uFF0D]/g, '-'); // various dashes -> '-'
    str = str.replace(/[\/\.]/g, '-'); // slash/dot -> '-'
    str = str.replace(/\s+/g, ' '); // normalize spaces

    // If string contains time or extra text, extract the leading date-like part
    // e.g. "2025-12-1T00:00:00" or "2025-12-1 (accepted)"
    const lead = str.match(/^(\d{4}[-\d\s]{0,20})/);
    if (lead) {
        // keep full string for further parse, but ensure we take first date-like token
        str = str.split(/[T\s(]/)[0];
    }

    // If just a year
    if (/^\d{4}$/.test(str)) {
        return new Date(parseInt(str, 10), 0, 1);
    }

    // Match year-month-day with 1-2 digits for month/day
    const m = str.match(/^(\d{4})[-\s]?(\d{1,2})[-\s]?(\d{1,2})$/);
    if (m) {
        const y = m[1];
        const mo = String(m[2]).padStart(2, '0');
        const d = String(m[3]).padStart(2, '0');
        const iso = `${y}-${mo}-${d}`;
        const dt = new Date(iso);
        if (!isNaN(dt.getTime())) return dt;
    }

    // Try loose match for cases like "2025 12 01" or "2025-12-01..."
    const loose = str.match(/(\d{4}).*?(\d{1,2}).*?(\d{1,2})/);
    if (loose) {
        const y = loose[1];
        const mo = String(loose[2]).padStart(2, '0');
        const d = String(loose[3]).padStart(2, '0');
        const iso = `${y}-${mo}-${d}`;
        const dt = new Date(iso);
        if (!isNaN(dt.getTime())) return dt;
    }

    // Last resort: Date.parse on the entire cleaned string
    const parsed = Date.parse(str);
    if (!isNaN(parsed)) return new Date(parsed);

    return null;
}

// Function to get the week range for a given date
function getWeekRange(date) {
    const d = new Date(date);
    const day = d.getDay(); // 0 (Sun) - 6 (Sat)
    const diffToMonday = (day === 0 ? -6 : 1) - day;

    const weekStart = new Date(d);
    weekStart.setDate(d.getDate() + diffToMonday);
    weekStart.setHours(0, 0, 0, 0);

    const weekEnd = new Date(weekStart);
    weekEnd.setDate(weekStart.getDate() + 6);
    weekEnd.setHours(23, 59, 59, 999);

    return { weekStart, weekEnd };
}

// Load CSV file
async function loadCSV() {
    try {
        // Set CSV file path
        const response = await fetch('data/vla_data.csv');
        const text = await response.text();
        
        // Parse CSV with Papa Parse
        const result = Papa.parse(text, {
            header: true,
            skipEmptyLines: true,
            dynamicTyping: true
        });
        
        if (result.errors.length > 0) {
            console.error('CSV parsing errors:', result.errors);
        }
        
        allData = result.data;
        
        // ------ Group rows by Abbreviation (preserve mapping info) ------
        const grouped = {}; // key: 略称 -> { baseRowFields..., challenges: Set, challengeOrder:[], subMap: Map, solveMap: Map }

        allData.forEach(row => {
            const key = (row['略称'] || '').trim();
            if (!key) return; // skip empty abbrev

            // Ensure row exists
            if (!grouped[key]) {
                grouped[key] = {
                    // keep first-seen basic fields (you can customize which field to keep/overwrite)
                    ...row,
                    challenges: new Set(),      // set of challenge tags
                    challengeOrder: [],         // keep insertion order for challenges
                    subMap: new Map(),          // Map subTag -> Set of challengeTags (sources)
                    solveMap: new Map()         // Map solveText -> Set of challengeTags (sources)
                };
            }

            const entry = grouped[key];

            // Parse challenge tags (split by ';' as your data uses)
            const challengeTags = (row['Challenge Tag'] || '')
                .split(';').map(s => s.trim()).filter(Boolean);

            // If no challengeTags but row has something else, fallback to single empty string to keep processing
            // (but we won't record empty challenge)
            if (challengeTags.length === 0) {
                // nothing to add to challenges
            } else {
                // add to entry.challenges preserving order
                challengeTags.forEach(ct => {
                    if (!entry.challenges.has(ct)) {
                        entry.challenges.add(ct);
                        entry.challengeOrder.push(ct);
                    }
                });
            }

            // Parse sub-challenge tags (split by ';')
            const subTags = (row['Sub-Challeng Tag'] || '')
                .split(';').map(s => s.trim()).filter(Boolean);

            // Map subTags -> challenge source:
            // If there are multiple challengeTags, assume positional pairing:
            // subTags[i] corresponds to challengeTags[i] if available, else fallback to first challengeTag.
            if (subTags.length > 0) {
                subTags.forEach((sub, idx) => {
                    let sourceChallenge = null;
                    if (challengeTags.length > idx) {
                        sourceChallenge = challengeTags[idx];
                    } else if (challengeTags.length === 1) {
                        sourceChallenge = challengeTags[0];
                    } else {
                        // fallback: if entry already has some challenges, use the first known one
                        sourceChallenge = entry.challengeOrder[0] || null;
                    }

                    if (!entry.subMap.has(sub)) entry.subMap.set(sub, new Set());
                    if (sourceChallenge) entry.subMap.get(sub).add(sourceChallenge);
                });
            }

            // Parse How to Solve (split by ',' as before)
            const solveItems = (row['How to Solve'] || '')
                .split(';').map(s => s.trim()).filter(Boolean);

            if (solveItems.length > 0) {
                solveItems.forEach((solve, idx) => {
                    // Pairing strategy similar to subTags:
                    let sourceChallenge = null;
                    if (challengeTags.length > idx) {
                        sourceChallenge = challengeTags[idx];
                    } else if (challengeTags.length === 1) {
                        sourceChallenge = challengeTags[0];
                    } else {
                        sourceChallenge = entry.challengeOrder[0] || null;
                    }

                    if (!entry.solveMap.has(solve)) entry.solveMap.set(solve, new Set());
                    if (sourceChallenge) entry.solveMap.get(solve).add(sourceChallenge);
                });
            }

            // Also keep other fields if needed (e.g., Year, URLs). We keep first row's fields by default.
            // If you want to merge other columns too, you can extend this block.
        });

        // Convert grouped back to allData array, and transform maps/sets to strings for compatibility
        allData = Object.keys(grouped).map(key => {
            const e = grouped[key];

            // produce merged 'Challenge Tag' string (preserve order)
            const mergedChallenge = e.challengeOrder.join(';');

            // For Sub-Challeng Tag and How to Solve we keep original string format (joined)
            // but we'll render using subMap/solveMap in the table, so storing the joined strings is optional.
            const mergedSub = [...e.subMap.keys()].join(';');
            const mergedSolve = [...e.solveMap.keys()].join(';');

            return {
                // use stored first-seen fields (like Year, Paper URL, Website URL, 略称)
                ...e,
                'Challenge Tag': mergedChallenge,
                'Sub-Challeng Tag': mergedSub,
                'How to Solve': mergedSolve,
                // expose the maps for rendering (we will use these maps later)
                __subMap: e.subMap,
                __solveMap: e.solveMap,
                __challengeOrder: e.challengeOrder
            };
        });


        // Filter out any empty rows that might have been created
        allData = allData.filter(row => row['略称'] && row['略称'].trim() !== '');
        
        allData.sort((a, b) => {
            const da = parseFlexibleDate(a["Year"]);
            const db = parseFlexibleDate(b["Year"]);
            return (db?.getTime() || 0) - (da?.getTime() || 0);
        });

        
        filteredData = [...allData];
        
        populateFilters();
        updateTable();
        updatePagination();
    } catch (error) {
        console.error('Error loading CSV:', error);
        document.getElementById('tableBody').innerHTML = 
            '<tr><td colspan="9" class="loading">Error loading CSV file. Please ensure vla_data.csv is in the data directory.</td></tr>';
    }
}

// Populate filter options
function populateFilters() {
    const uniqueValues = (field) => {
        const separator = (field === 'Challenge Tag' || field === 'Sub-Challeng Tag' || field === 'Dataset' || field === 'Evaluation') ? /;/ : /,/;
        return [...new Set(allData.flatMap(row =>
            (row[field] || '').split(separator).map(value => value.trim()).filter(Boolean)
        ))];
    };
    
    const challengeTags = uniqueValues('Challenge Tag');
    const subChallengeTags = uniqueValues('Sub-Challeng Tag');
    const trainingTypes = uniqueValues('Training Type');
    const datasets = uniqueValues('Dataset');
    const evaluations = uniqueValues('Evaluation');
    
    populateSelect('challengeFilter', challengeTags);
    populateSelect('subChallengeFilter', subChallengeTags);
    populateSelect('trainingFilter', trainingTypes);
    populateSelect('datasetFilter', datasets);
    populateSelect('evaluationFilter', evaluations);
}

function populateSelect(selectId, options) {
    const select = document.getElementById(selectId);
    select.innerHTML = '<option value="">All</option>';
    options.sort().forEach(option => {
        const optionElement = document.createElement('option');
        optionElement.value = option;
        optionElement.textContent = option;
        select.appendChild(optionElement);
    });
}
// Generate a stable HSL color based on a string
function stringToColor(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }

    const hue = Math.abs(hash) % 360; // 0–359
    const saturation = 55; // moderate saturation
    const lightness = 68; // soft appearance

    return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

// Generate tag class function
function getTagClass(type, value) {
    if (!value) return 'tag tag-default';
    const lowerValue = value.toLowerCase().trim();
    
    if (type === 'challenge') {
        if (lowerValue.includes('fusion') || lowerValue.includes('representation')) return 'tag tag-challenge-fusion';
        if (lowerValue.includes('execution') || lowerValue.includes('complex')) return 'tag tag-challenge-execution';
        if (lowerValue.includes('generalization') || lowerValue.includes('learning')) return 'tag tag-challenge-generalization';
        if (lowerValue.includes('security') || lowerValue.includes('reliable')) return 'tag tag-challenge-security';
        if (lowerValue.includes('dataset') || lowerValue.includes('benchmarking')) return 'tag tag-challenge-dataset';
    }
    if (type === 'dataset') {
        // return `tag tag-auto" style="background-color:${stringToColor(value)}; color:#000;`;
        return `tag tag-auto" style="background-color:#1abc9c; color:#000;`;
    }
    if (type === 'evaluation') {
        // return `tag tag-auto" style="background-color:${stringToColor(value)}; color:#000;`;
        return `tag tag-auto" style="background-color:#90b9e4; color:#000;`;
    }
    if (type === 'dataset-eval') {
        return 'tag tag-dataset-eval';
    }
    if (type === 'training') {
        return 'tag tag-default';
    }
    if (type === 'default') {
        return 'tag tag-default';
    }
    
    const normalizedValue = value.toLowerCase().replace(/[\s-]/g, '-');
    return `tag tag-${type}-${normalizedValue}`;
}

// Highlight search term function
function highlightSearchTerm(text, searchTerm) {
    if (!text || !searchTerm || searchTerm.length < 2) return text;
    
    // Escape HTML entities
    const escapeHtml = (str) => {
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    };
    
    // Escape regex special characters
    const escapeRegExp = (str) => {
        return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    };
    
    const safeText = escapeHtml(text);
    const regex = new RegExp(`(${escapeRegExp(searchTerm)})`, 'gi');
    return safeText.replace(regex, '<mark>$1</mark>');
}

// Update table
function updateTable() {
    const tbody = document.getElementById('tableBody');
    tbody.innerHTML = '';
    
    const searchTerm = document.getElementById('searchInput').value.trim();
    const start = (currentPage - 1) * recordsPerPage;
    const end = start + recordsPerPage;
    const pageData = filteredData.slice(start, end);
    
    if (pageData.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" style="text-align: center;">No data found</td></tr>';
        return;
    }
    
    pageData.forEach(row => {
        const tr = document.createElement('tr');
        const renderTagCell = (field, type, separator = ',') => {
            const values = (row[field] || '').split(separator).map(value => value.trim()).filter(Boolean);
            if (values.length === 0) return '-';
            return values.map((value) => {
                const highlighted = searchTerm ? highlightSearchTerm(value, searchTerm) : value;
                return `<span class="${getTagClass(type, value)}">${highlighted}</span>`;
            }).join(' ');
        };
        
        // Abbreviation
        const abbreviation = row['略称'] || '';
        const abbreviationContent = abbreviation ? (searchTerm ? highlightSearchTerm(abbreviation, searchTerm) : abbreviation) : '-';
        tr.innerHTML += `<td><strong>${abbreviationContent}</strong></td>`;
        
        // Year
        const yearValue = row['Year'] || '';
        const yearContent = yearValue ? (searchTerm ? highlightSearchTerm(String(yearValue), searchTerm) : yearValue) : '-';
        tr.innerHTML += `<td>${yearContent}</td>`;
        
        // Links
        const paperUrl = row['Paper URL'] || '';
        const websiteUrl = row['Website URL'] || '';
        let linkContent = '';
        if (paperUrl) {
            linkContent += `<a href="${paperUrl}" target="_blank" class="link mr-2">Paper</a>`;
        }
        if (websiteUrl) {
            linkContent += `<a href="${websiteUrl}" target="_blank" class="link">Website</a>`;
        }
        tr.innerHTML += `<td>${linkContent || '-'}</td>`;
        
        // Challenge Tag & Sub-Challeng Tag (paired bubbles)
        // const rawChallenge = row['Challenge Tag'] || '';
        // const rawSubChallenge = row['Sub-Challeng Tag'] || '';
        // const challengeTags = rawChallenge.split(';').map(v => v.trim()).filter(Boolean);
        // const subChallenges = rawSubChallenge.split(';').map(v => v.trim()).filter(Boolean);
        
        // --- prepare challenge tags (display once, ordered) ---
        const challengeTags = (row['Challenge Tag'] || '').split(';').map(v=>v.trim()).filter(Boolean);
        let challengeHtml = '-';
        if (challengeTags.length > 0) {
            challengeHtml = challengeTags.map(tag => {
                const highlighted = searchTerm ? highlightSearchTerm(tag, searchTerm) : tag;
                const cls = getTagClass('challenge', tag);
                return `<span class="${cls}">${highlighted}</span>`;
            }).join(' ');
        }
        tr.innerHTML += `<td>${challengeHtml}</td>`;

        // --- sub-challenge: render each sub possibly multiple times (one per source challenge) ---
        let subHtml = '-';
        const subMap = row.__subMap || new Map();
        if (subMap.size > 0) {
            const parts = [];
            // iterate keys in insertion order
            for (const [sub, challengeSet] of subMap.entries()) {
                // For each source challenge that produced this sub, create a colored span.
                // If a sub came from multiple challenges, we'll show multiple spans (same text, diff colors)
                for (const ch of challengeSet) {
                    const highlighted = searchTerm ? highlightSearchTerm(sub, searchTerm) : sub;
                    const cls = getTagClass('challenge', ch || '');
                    parts.push(`<span class="${cls}">${highlighted}</span>`);
                }
            }
            if (parts.length > 0) subHtml = parts.join(' ');
        }
        tr.innerHTML += `<td>${subHtml}</td>`;
        
        // --- How to Solve: similar to sub ---
        // We'll use row.__solveMap (solveText -> Set of associated challenge tags)
        let solveHtml = '-';
        const solveMap = row.__solveMap || new Map();
        if (solveMap.size > 0) {
            const parts = [];
            for (const [solve, challengeSet] of solveMap.entries()) {
                for (const ch of challengeSet) {
                    // Use challenge color for how-to-solve that originated with that challenge
                    const highlighted = searchTerm ? highlightSearchTerm(solve, searchTerm) : solve;
                    // we reuse challenge coloring so use getTagClass('challenge', ch)
                    const cls = getTagClass('challenge', ch || '');
                    parts.push(`<span class="${cls}">${highlighted}</span>`);
                }
            }
            if (parts.length > 0) solveHtml = parts.join(' ');
        }
        tr.innerHTML += `<td>${solveHtml}</td>`;
        
        // Training Type
        tr.innerHTML += `<td>${renderTagCell('Training Type', 'training')}</td>`;
        
        // Dataset (use ';' and single neutral bubble color)
        tr.innerHTML += `<td>${renderTagCell('Dataset', 'dataset', ';')}</td>`;
        
        // Evaluation (use ';' and single neutral bubble color)
        tr.innerHTML += `<td>${renderTagCell('Evaluation', 'evaluation', ';')}</td>`;
        
        tbody.appendChild(tr);
    });
}

// Calculate search score
function calculateSearchScore(searchTerm, row) {
    const fieldWeights = {
        '略称': 10,
        'Year': 6,
        'Challenge Tag': 8,
        'Sub-Challeng Tag': 7,
        'How to Solve': 7,
        'Training Type': 5,
        'Dataset': 4,
        'Evaluation': 4
    };
    
    let score = 0;
    const searchLower = searchTerm.toLowerCase();
    
    for (const [field, value] of Object.entries(row)) {
        if (!value) continue;
        
        const valueLower = String(value).toLowerCase();
        const weight = fieldWeights[field] || 1;
        
        if (valueLower === searchLower) {
            score += weight * 10;
        } else if (valueLower.split(/\s+/).some(word => word.startsWith(searchLower))) {
            score += weight * 5;
        } else if (valueLower.includes(searchLower)) {
            score += weight * 2;
        }
    }
    
    return score;
}


function applyFilters() {
    const searchTerm = document.getElementById('searchInput').value.toLowerCase();
    const challengeFilter = document.getElementById('challengeFilter').value;
    const subChallengeFilter = document.getElementById('subChallengeFilter').value;
    const trainingFilter = document.getElementById('trainingFilter').value;
    const datasetFilter = document.getElementById('datasetFilter').value;
    const evaluationFilter = document.getElementById('evaluationFilter').value;
    
    const results = allData.map(row => {
        const includesValue = (value, target, field) => {
            const separator = (field === 'Challenge Tag' || field === 'Sub-Challeng Tag' || field === 'Dataset' || field === 'Evaluation')
                ? /;/
                : /,/;

            const values = (value || '').split(separator).map(v => v.trim()).filter(Boolean);

            // Dataset / Evaluation: allow partial match, e.g. "CLIPORT" matches "CLIPORT;VLMA-BENCH"
            if (field === 'Dataset' || field === 'Evaluation') {
                return values.some(v => v.toLowerCase().includes(target.toLowerCase()));
            }

            // Other fields: keep strict equality
            return values.includes(target);
        };

        
        if (challengeFilter && !includesValue(row['Challenge Tag'], challengeFilter, 'Challenge Tag')) return null;
        if (subChallengeFilter && !includesValue(row['Sub-Challeng Tag'], subChallengeFilter, 'Sub-Challeng Tag')) return null;
        if (trainingFilter && !includesValue(row['Training Type'], trainingFilter, 'Training Type')) return null;
        if (datasetFilter && !includesValue(row['Dataset'], datasetFilter, 'Dataset')) return null;
        if (evaluationFilter && !includesValue(row['Evaluation'], evaluationFilter, 'Evaluation')) return null;

        // Search term matching and score calculation
        if (searchTerm) {
            const score = calculateSearchScore(searchTerm, row);
            if (score === 0) {
                // If score is 0, perform general search
                const searchMatch = Object.values(row).some(value => 
                    String(value).toLowerCase().includes(searchTerm)
                );
                if (!searchMatch) return null;
            }
            return { row, score };
        }
        
        return { row, score: 0 };
    }).filter(item => item !== null);
    

    if (searchTerm && results.some(item => item.score > 0)) {
        results.sort((a, b) => b.score - a.score);
    }
    
    filteredData = results.map(item => item.row);
    
    currentPage = 1;
    updateTable();
    updatePagination();
}


function sortTable(column) {
    if (sortColumn === column) {
        sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
    } else {
        sortColumn = column;
        sortDirection = 'asc';
    }
    
    const columnMap = {
        0: '略称',
        1: 'Year',
        3: 'Challenge Tag',
        4: 'Sub-Challeng Tag',
        5: 'How to Solve',
        6: 'Training Type',
        7: 'Dataset',
        8: 'Evaluation'
    };
    
    const key = columnMap[column];
    
    filteredData.sort((a, b) => {
        let aVal = a[key] || '';
        let bVal = b[key] || '';

        // Special case: Year sorting using real dates
        if (key === 'Year') {
            const da = parseFlexibleDate(aVal);
            const db = parseFlexibleDate(bVal);
            if (sortDirection === 'asc') {
                return (da?.getTime() || 0) - (db?.getTime() || 0);
            } else {
                return (db?.getTime() || 0) - (da?.getTime() || 0);
            }
        }

        // Non-Year columns: keep original sorting rule
        if (['Challenge Tag', 'Sub-Challeng Tag', 'How to Solve', 'Training Type', 'Dataset', 'Evaluation'].includes(key)) {
            aVal = aVal.split(',')[0].trim() || '';
            bVal = bVal.split(',')[0].trim() || '';
        }
        if (sortDirection === 'asc') {
            return aVal.localeCompare(bVal);
        } else {
            return bVal.localeCompare(aVal);
        }
    });
    
    // Update sort indicators
    document.querySelectorAll('th.sortable').forEach(th => {
        th.classList.remove('sort-asc', 'sort-desc');
    });
    
    const currentTh = document.querySelector(`th[data-column="${column}"]`);
    if (currentTh) {
        currentTh.classList.add(sortDirection === 'asc' ? 'sort-asc' : 'sort-desc');
    }
    
    updateTable();
}

function updatePagination() {
    const totalRecords = filteredData.length;
    const totalPages = Math.ceil(totalRecords / recordsPerPage);
    const start = totalRecords > 0 ? (currentPage - 1) * recordsPerPage + 1 : 0;
    const end = Math.min(currentPage * recordsPerPage, totalRecords);
    
    document.getElementById('startRecord').textContent = start;
    document.getElementById('endRecord').textContent = end;
    document.getElementById('totalRecords').textContent = totalRecords;
    document.getElementById('currentPage').textContent = currentPage;
    document.getElementById('totalPages').textContent = totalPages || 1;
    
    document.getElementById('firstPage').disabled = currentPage === 1;
    document.getElementById('prevPage').disabled = currentPage === 1;
    document.getElementById('nextPage').disabled = currentPage >= totalPages;
    document.getElementById('lastPage').disabled = currentPage >= totalPages;
}

// Change page
function changePage(newPage) {
    const totalPages = Math.ceil(filteredData.length / recordsPerPage);
    if (newPage >= 1 && newPage <= totalPages) {
        currentPage = newPage;
        updateTable();
        updatePagination();
    }
}

// Clear all filters
function clearAllFilters() {
    document.getElementById('searchInput').value = '';
    document.getElementById('challengeFilter').value = '';
    document.getElementById('subChallengeFilter').value = '';
    document.getElementById('trainingFilter').value = '';
    document.getElementById('datasetFilter').value = '';
    document.getElementById('evaluationFilter').value = '';
    applyFilters();
}

// Load latest.csv and render horizontally scrollable cards
async function loadLatestCSV() {
    try {
        const response = await fetch('data/latest.csv');
        const text = await response.text();

        const result = Papa.parse(text, {
            header: true,
            skipEmptyLines: true,
            dynamicTyping: true
        });

        let data = result.data.filter(row => row["略称"]?.trim());

        // --- reuse same grouping logic as vla_data.csv ---
        const grouped = {};

        data.forEach(row => {
            const key = (row['略称'] || '').trim();
            if (!key) return;

            if (!grouped[key]) {
                grouped[key] = {
                    ...row,
                    challenges: new Set(),
                    challengeOrder: [],
                    subMap: new Map(),
                    solveMap: new Map()
                };
            }

            const entry = grouped[key];

            const challengeTags = (row['Challenge Tag'] || '')
                .split(';').map(s => s.trim()).filter(Boolean);
            challengeTags.forEach(ct => {
                if (!entry.challenges.has(ct)) {
                    entry.challenges.add(ct);
                    entry.challengeOrder.push(ct);
                }
            });

            const subTags = (row['Sub-Challeng Tag'] || '')
                .split(';').map(s => s.trim()).filter(Boolean);
            subTags.forEach((sub, idx) => {
                let src = challengeTags[idx] || challengeTags[0] || null;
                if (!entry.subMap.has(sub)) entry.subMap.set(sub, new Set());
                if (src) entry.subMap.get(sub).add(src);
            });

            const solveItems = (row['How to Solve'] || '')
                .split(';').map(s => s.trim()).filter(Boolean);
            solveItems.forEach((solve, idx) => {
                let src = challengeTags[idx] || challengeTags[0] || null;
                if (!entry.solveMap.has(solve)) entry.solveMap.set(solve, new Set());
                if (src) entry.solveMap.get(solve).add(src);
            });
        });

        data = Object.keys(grouped).map(k => {
            const e = grouped[k];
            return {
                ...e,
                __challengeOrder: e.challengeOrder,
                __subMap: e.subMap,
                __solveMap: e.solveMap,
                'Challenge Tag': e.challengeOrder.join(';'),
                'Sub-Challeng Tag': [...e.subMap.keys()].join(';'),
                'How to Solve': [...e.solveMap.keys()].join(';')
            };
        });

        // Sort newest Year first
        data.sort((a, b) => {
            const da = parseFlexibleDate(a["Year"]);
            const db = parseFlexibleDate(b["Year"]);
            return (db?.getTime() || 0) - (da?.getTime() || 0);
        });

        latestAllData = data;
        applyLatestWeekFilter();
            
    } catch (err) {
        console.error("Error loading latest.csv:", err);
    }
}

// Apply latest week filter and render cards
function applyLatestWeekFilter() {
    const select = document.getElementById("latestRangeSelect");
    const offset = parseInt(select.value, 10) - 1; // 0 = most recent week

    if (!latestAllData.length) return;

    // 1. 找到最新的 Updated Date
    const allDates = latestAllData
        .map(r => parseFlexibleDate(r["Updated Date"]))
        .filter(Boolean);

    if (!allDates.length) return;

    const latestDate = new Date(Math.max(...allDates.map(d => d.getTime())));

    // 2. 计算目标周（向前 offset 周）
    const baseWeek = getWeekRange(latestDate);
    const targetWeekStart = new Date(baseWeek.weekStart);
    targetWeekStart.setDate(baseWeek.weekStart.getDate() - offset * 7);

    const targetWeekEnd = new Date(targetWeekStart);
    targetWeekEnd.setDate(targetWeekStart.getDate() + 6);
    targetWeekEnd.setHours(23, 59, 59, 999);

    // 3. 严格筛选：Updated Date ∈ 该周
    const filtered = latestAllData.filter(row => {
        const d = parseFlexibleDate(row["Updated Date"]);
        return d && d >= targetWeekStart && d <= targetWeekEnd;
    });

    // 4. 更新卡片
    renderLatestCards(filtered);

    // 5. 更新 Updated on 文案（该周的最大 Updated Date）
    updateLatestUpdatedDate(filtered, targetWeekEnd);
}

// Update "Updated on" text for latest cards
function updateLatestUpdatedDate(dataInWeek, fallbackDate) {
    const span = document.getElementById("latestUpdatedDate");
    if (!span) return;

    if (!dataInWeek.length) {
        span.innerHTML = `
        <div class="latest-empty">
            <div class="latest-empty-title">No updates this week</div>
            <div class="latest-empty-sub">
                No papers were added during the selected week.
            </div>
        </div>
    `;
        return;
    }

    const dates = dataInWeek
        .map(r => parseFlexibleDate(r["Updated Date"]))
        .filter(Boolean);

    const maxDate = dates.length
        ? new Date(Math.max(...dates.map(d => d.getTime())))
        : fallbackDate;

    const yyyy = maxDate.getFullYear();
    const mm = String(maxDate.getMonth() + 1).padStart(2, '0');
    const dd = String(maxDate.getDate()).padStart(2, '0');

    span.textContent = `(Updated on: ${yyyy}-${mm}-${dd})`;
}


// Render card layout using SAME logic as vla_data table
function renderLatestCards(latestData) {
    const container = document.getElementById("latestScroll");
    container.innerHTML = "";

    latestData.forEach(row => {
        const card = document.createElement("div");
        card.className = "latest-card";

        // ----- challenge tags -----
        const challengeHTML = row.__challengeOrder.length
            ? row.__challengeOrder.map(tag => {
                return `<span class="${getTagClass('challenge', tag)}">${tag}</span>`;
            }).join(' ')
            : "-";

        // ----- sub-challenge -----
        let subHTML = "-";
        if (row.__subMap.size > 0) {
            const arr = [];
            for (const [sub, challengeSet] of row.__subMap.entries()) {
                for (const ch of challengeSet) {
                    arr.push(`<span class="${getTagClass('challenge', ch)}">${sub}</span>`);
                }
            }
            subHTML = arr.join(" ");
        }

        // ----- how to solve -----
        let solveHTML = "-";
        if (row.__solveMap.size > 0) {
            const arr = [];
            for (const [solve, challengeSet] of row.__solveMap.entries()) {
                for (const ch of challengeSet) {
                    arr.push(`<span class="${getTagClass('challenge', ch)}">${solve}</span>`);
                }
            }
            solveHTML = arr.join(" ");
        }

        // ----- dataset -----
        const datasetHTML = (row["Dataset"] || "")
            .split(';').filter(Boolean)
            .map(d => `<span class="${getTagClass('dataset', d)}">${d}</span>`)
            .join(" ") || "-";

        // ----- evaluation -----
        const evalHTML = (row["Evaluation"] || "")
            .split(';').filter(Boolean)
            .map(d => `<span class="${getTagClass('evaluation', d)}">${d}</span>`)
            .join(" ") || "-";

        // ----- Links -----
        let linkHTML = "-";
        if (row["Paper URL"] || row["Website URL"]) {
            linkHTML = "";
            if (row["Paper URL"]) {
                linkHTML += `<a href="${row["Paper URL"]}" target="_blank" class="mr-2">Paper</a>`;
            }
            if (row["Website URL"]) {
                linkHTML += `<a href="${row["Website URL"]}" target="_blank">Website</a>`;
            }
        }

        // ----- Training Type -----
        let trainingHTML = "-";
        if (row["Training Type"]) {
            trainingHTML = row["Training Type"]
                .split(',')
                .map(t => `<span class="${getTagClass('training', t.trim())}">${t.trim()}</span>`)
                .join(" ");
        }

        card.innerHTML = `
            <div class="latest-title">${row["略称"]}</div>
            <div class="latest-subtitle">${row["Title"] || ""}</div>

            <div class="latest-section-label mt-2">Date</div>
            <div>${row["Year"] || "-"}</div>

            <div class="latest-section-label">Challenge</div>
            <div>${challengeHTML}</div>

            <div class="latest-section-label">Sub-Challenge</div>
            <div>${subHTML}</div>

            <div class="latest-section-label">How to Solve</div>
            <div>${solveHTML}</div>

            <div class="latest-section-label">Training Type</div>
            <div>${trainingHTML}</div>

            <div class="latest-section-label">Dataset</div>
            <div>${datasetHTML}</div>

            <div class="latest-section-label">Evaluation</div>
            <div>${evalHTML}</div>

            <div class="latest-section-label mt-2">Link</div>
            <div>${linkHTML}</div>
        `;


        container.appendChild(card);
    });
}


// Add call after DOM loaded
document.addEventListener("DOMContentLoaded", function () {
    loadLatestCSV();
});

// Apply latest range filter when select changes
document.getElementById("latestRangeSelect")
    ?.addEventListener("change", applyLatestWeekFilter);

// Set up event listeners
document.addEventListener('DOMContentLoaded', function() {
    loadCSV();

    // Filter events
    document.getElementById('searchInput').addEventListener('input', applyFilters);
    document.getElementById('challengeFilter').addEventListener('change', applyFilters);
    document.getElementById('subChallengeFilter').addEventListener('change', applyFilters);
    document.getElementById('trainingFilter').addEventListener('change', applyFilters);
    document.getElementById('datasetFilter').addEventListener('change', applyFilters);
    document.getElementById('evaluationFilter').addEventListener('change', applyFilters);
    
    // Sort events
    document.querySelectorAll('th.sortable').forEach(th => {
        th.addEventListener('click', () => sortTable(parseInt(th.dataset.column)));
    });
    
    // Pagination events
    document.getElementById('firstPage').addEventListener('click', () => changePage(1));
    document.getElementById('prevPage').addEventListener('click', () => changePage(currentPage - 1));
    document.getElementById('nextPage').addEventListener('click', () => changePage(currentPage + 1));
    document.getElementById('lastPage').addEventListener('click', () => {
        const totalPages = Math.ceil(filteredData.length / recordsPerPage);
        changePage(totalPages);
    });
    
    document.getElementById('recordsPerPage').addEventListener('change', (e) => {
        recordsPerPage = parseInt(e.target.value);
        currentPage = 1;
        updateTable();
        updatePagination();
    });
});