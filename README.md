# Admin Form Builder for Salesforce Lightning Platform

A drag-and-drop form builder Lightning Web Component (LWC) for Salesforce administrators. This solution allows admins to visually create dynamic forms and save submissions to custom Salesforce objects.

## Features

- **Drag & Drop Interface**: Easily build forms by dragging elements from the palette onto the canvas
- **Multiple Field Types**: Support for various input types including:
  - Text Input
  - Email
  - Number
  - Date
  - Phone
  - Text Area
  - Checkbox
  - Picklist (Dropdown)
  - Radio Buttons
  - Section Headers
  - Rich Text
  - File Upload
- **Form Preview**: Preview your form before publishing
- **Element Properties**: Configure labels, placeholders, default values, and validation rules
- **Required Fields**: Mark fields as mandatory
- **Reorder Elements**: Drag elements within the canvas to reorder
- **Form Persistence**: Save form definitions to custom Salesforce objects
- **Form Submissions**: Capture and store user submissions
- **Submission Viewer**: View and manage form submissions

## Components

### 1. Form Builder (`formBuilder`)
The main admin interface for creating and editing forms.

**Features:**
- Drag and drop form element palette
- Visual form canvas
- Properties panel for configuring elements
- Save, preview, and new form actions

### 2. Form Renderer (`formRenderer`)
Renders saved forms for end-user submissions.

**Properties:**
- `formDefinitionId`: The ID of the form to render
- `previewMode`: Set to true for preview-only mode

### 3. Form Submission Viewer (`formSubmissionViewer`)
Admin component for viewing and managing form submissions.

**Features:**
- Form selector dropdown
- Submissions data table
- Detailed submission view
- Status management

## Custom Objects

### Form_Definition__c
Stores form configurations and definitions.

| Field | Type | Description |
|-------|------|-------------|
| Name | Text | Form name |
| Description__c | Long Text | Form description |
| Form_Definition__c | Long Text | JSON form definition |
| Is_Active__c | Checkbox | Whether the form is active |
| Version__c | Number | Form version number |
| Last_Modified_Date__c | DateTime | Last modification timestamp |

### Form_Submission__c
Stores user form submissions.

| Field | Type | Description |
|-------|------|-------------|
| Name | Auto Number | Submission reference number |
| Form_Definition__c | Lookup | Related form definition |
| Submission_Data__c | Long Text | JSON submission data |
| Submitted_Date__c | DateTime | Submission timestamp |
| Status__c | Picklist | Submission status |

## Installation

### Prerequisites
- Salesforce org with Lightning Experience enabled
- System Administrator profile or equivalent permissions
- Salesforce CLI (for deployment)

### Deploy to Salesforce

1. Clone this repository:
```bash
git clone <repository-url>
cd admin-form-builder
```

2. Authorize your Salesforce org:
```bash
sf org login web -a MyOrg
```

3. Deploy the metadata:
```bash
sf project deploy start -o MyOrg
```

4. Assign the permission set to users:
```bash
sf org assign permset -n Form_Builder_Admin -o MyOrg
```

## Usage

### Creating a Form

1. Navigate to the Form Builder component (add to an App Page, Home Page, or Lightning Tab)
2. Enter a form name and description
3. Drag elements from the left panel onto the canvas
4. Click on elements to configure their properties:
   - Label
   - API Name
   - Placeholder
   - Required
   - Options (for picklists and radio buttons)
   - Help Text
   - Validation (min/max length)
5. Reorder elements by dragging within the canvas
6. Click "Preview" to see how the form will look
7. Click "Save Form" to persist the form definition

### Rendering a Form

Add the `formRenderer` component to a Lightning page and configure:
- Set `formDefinitionId` to the ID of the form to display
- Or use on a Form_Definition__c record page for automatic context

### Viewing Submissions

1. Navigate to the Form Submission Viewer component
2. Select a form from the dropdown
3. View submissions in the data table
4. Click "View Details" to see full submission data

## Form Definition JSON Structure

```json
{
  "name": "Contact Form",
  "description": "Customer contact information",
  "elements": [
    {
      "id": "element_1234567890_abc123",
      "type": "text",
      "label": "First Name",
      "apiName": "first_name",
      "placeholder": "Enter your first name",
      "required": true,
      "helpText": "Your legal first name"
    },
    {
      "id": "element_1234567891_def456",
      "type": "email",
      "label": "Email Address",
      "apiName": "email",
      "placeholder": "example@email.com",
      "required": true
    },
    {
      "id": "element_1234567892_ghi789",
      "type": "picklist",
      "label": "Department",
      "apiName": "department",
      "options": [
        { "label": "Sales", "value": "sales" },
        { "label": "Support", "value": "support" },
        { "label": "Engineering", "value": "engineering" }
      ],
      "required": false
    }
  ]
}
```

## Available Element Types

| Type | Description | Properties |
|------|-------------|------------|
| `text` | Single-line text input | label, apiName, placeholder, required, defaultValue, helpText, minLength, maxLength |
| `email` | Email input with validation | label, apiName, placeholder, required, helpText |
| `number` | Numeric input | label, apiName, placeholder, required, defaultValue, helpText |
| `date` | Date picker | label, apiName, required, helpText |
| `phone` | Phone number input | label, apiName, placeholder, required, helpText |
| `textarea` | Multi-line text input | label, apiName, placeholder, required, helpText, maxLength |
| `checkbox` | Single checkbox | label, apiName, required, helpText |
| `picklist` | Dropdown select | label, apiName, placeholder, required, options, helpText |
| `radio` | Radio button group | label, apiName, required, options |
| `section` | Section header | label |
| `richtext` | Rich text editor | label, apiName, helpText |
| `file` | File upload | label, apiName, required, helpText |

## Security Considerations

- All Apex methods use `WITH SECURITY_ENFORCED` for CRUD/FLS enforcement
- Permission sets control access to custom objects and fields
- Form submissions are validated server-side
- JSON definitions are validated before storage

## Apex Controller Methods

| Method | Description |
|--------|-------------|
| `saveFormDefinition` | Create or update a form definition |
| `getFormDefinition` | Retrieve a form definition by ID |
| `getActiveFormDefinitions` | Get all active form definitions |
| `submitFormData` | Submit form data |
| `getFormSubmissions` | Get submissions for a form |
| `getFormSubmission` | Get a single submission |
| `deleteFormDefinition` | Delete a form and its submissions |
| `updateFormStatus` | Update form active status |
| `duplicateForm` | Create a copy of a form |

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
