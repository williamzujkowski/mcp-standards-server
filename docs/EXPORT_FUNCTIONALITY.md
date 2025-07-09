# Export Functionality Documentation

## Overview

The MCP Standards Server Web UI now includes comprehensive export functionality that allows users to export standards in various formats. This feature supports both individual and bulk exports with proper error handling and user feedback.

## Features

### 1. Export Formats

- **JSON (Single file)**: Exports selected standards or all standards into a single JSON file with metadata
- **Markdown (Individual files)**: Exports each standard as a separate Markdown file
- **JSON (Individual files)**: Exports each standard as a separate JSON file

### 2. Export Modes

#### Individual Export
- Quick export button on each standard card
- Exports a single standard in Markdown format by default
- Available directly from the standards browser

#### Bulk Export
- Select multiple standards using the "Select Multiple" mode
- Export all selected standards at once
- Option to export all standards if none are selected
- Progress indicators during export process

### 3. User Interface

#### Selection Mode
- Toggle "Select Multiple" button to enable selection mode
- Checkboxes appear on each standard card
- "Select All" and "Clear" buttons for quick selection management
- Selection count displayed in the header

#### Export Dialog
- Clean dialog interface for choosing export format
- Clear description of what will be exported
- Radio buttons for format selection
- Export button with loading state

#### Feedback
- Success notifications via snackbar
- Error alerts for failed exports
- Progress indication during export

## Implementation Details

### Frontend Components

#### StandardsBrowser.tsx
- Added state management for selection and export
- Implemented export dialog with format options
- Added checkbox UI for multi-select mode
- Error handling and user feedback

#### StandardsService.ts
- `exportStandard()`: Exports a single standard
- `exportBulkStandards()`: Exports multiple standards efficiently

### Backend Endpoints

#### GET `/api/export/{standard_id}`
- Exports a single standard
- Supports `markdown` and `json` formats
- Returns file download response

#### POST `/api/export/bulk`
- Exports multiple standards in a single request
- Request body: `{ standards: string[], format: string }`
- Returns combined JSON file

## Usage

### Exporting a Single Standard

1. Navigate to the Standards Browser
2. Find the standard you want to export
3. Click the "Export" button on the standard card
4. The standard will be downloaded as a Markdown file

### Bulk Exporting Standards

1. Click "Select Multiple" in the header
2. Check the standards you want to export
3. Click "Export Standards"
4. Choose your export format:
   - JSON (Single file): All standards in one JSON file
   - Markdown (Individual files): Each standard as a separate .md file
   - JSON (Individual files): Each standard as a separate .json file
5. Click "Export" and wait for the download(s)

### Exporting All Standards

1. Click "Export Standards" without selecting any
2. Choose your export format
3. All standards will be exported

## Testing

A test script is provided at `test_export_functionality.py` that verifies:
- Single standard export (Markdown and JSON)
- Bulk export with specific standards
- Bulk export of all standards
- Error handling

Run the test with:
```bash
python test_export_functionality.py
```

## Error Handling

The export functionality includes comprehensive error handling:
- Network errors are caught and displayed to the user
- Invalid format requests return appropriate error messages
- Failed exports show error notifications
- The UI remains responsive during exports

## Performance Considerations

- Individual file exports include a 100ms delay between files to prevent browser overwhelm
- Bulk JSON export uses a dedicated endpoint for efficiency
- Large exports show progress indicators
- Temporary files are created on the server and cleaned up after download