# LWC Form Builder

A powerful drag-and-drop form builder for Salesforce Lightning Experience. Administrators can create custom forms without code, and users can fill them out with data saved to Salesforce objects.

## Features

### Form Builder (Admin)
- **Drag & Drop Interface**: Easily build forms by dragging elements onto the canvas
- **Multiple Field Types**: Text, Email, Phone, Number, Date, DateTime, Textarea, Checkbox, Picklist, Radio, Multi-Select, File Upload
- **Layout Elements**: Section headers and help text blocks for better form organization
- **Field Properties**: Customize labels, placeholders, help text, required validation, and default values
- **Object Mapping**: Optionally map form fields to Salesforce object fields for automatic record creation
- **Form Management**: Save, load, duplicate, and delete form definitions
- **Live Preview**: Preview forms before publishing

### Form Renderer (Users)
- **Responsive Design**: Works on desktop and mobile devices
- **Client-Side Validation**: Required field validation before submission
- **Success Feedback**: Clear confirmation after form submission
- **Reset Functionality**: Allow users to start over

### Form Response Viewer (Admin)
- **Response Analytics**: View total responses, today's count, and weekly stats
- **Sortable Data Table**: Sort responses by submission date, submitter, etc.
- **Detailed View**: Expand any response to see all submitted values
- **Target Record Links**: Quick access to records created from form submissions

## Installation

### Prerequisites
- Salesforce CLI (sf or sfdx)
- A Salesforce org (Developer Edition, Sandbox, or Scratch Org)

### Deploy to Salesforce

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd lwc-form-builder
   ```

2. **Authenticate with your org**
   ```bash
   sf org login web -a MyOrg
   ```

3. **Deploy the metadata**
   ```bash
   sf project deploy start -o MyOrg
   ```

4. **Assign permission sets**
   ```bash
   # For administrators who will create forms
   sf org assign permset -n Form_Builder_Admin -o MyOrg
   
   # For users who will fill out forms
   sf org assign permset -n Form_Builder_User -o MyOrg
   ```

## Usage

### Creating a Form (Admin)

1. Navigate to the **Form Builder** tab
2. Enter form settings:
   - Form Name (required)
   - Description (optional)
   - Target Object (optional - for automatic record creation)
   - Active status
3. Drag form elements from the left palette onto the canvas
4. Click on a field to edit its properties in the right panel:
   - Label
   - API Name
   - Placeholder
   - Help Text
   - Required
   - Default Value
   - Target Field Mapping (if target object is selected)
5. Reorder fields by dragging them within the canvas
6. Click **Preview** to test the form
7. Click **Save** to save the form definition

### Filling Out Forms (Users)

1. Navigate to the **Fill Forms** tab
2. Click on any available form card
3. Complete all required fields
4. Click **Submit**

### Viewing Responses (Admin)

1. Navigate to the **Form Responses** tab
2. Select a form from the dropdown
3. View response statistics and data table
4. Click on a row to see detailed response data

## Custom Objects

### Form_Definition__c
Stores form configurations and structure.

| Field | Type | Description |
|-------|------|-------------|
| Name | Text | Form name |
| Description__c | Long Text Area | Form description |
| Form_JSON__c | Long Text Area | JSON structure of form fields |
| Target_Object__c | Text | API name of target object |
| Is_Active__c | Checkbox | Whether form accepts submissions |

### Form_Response__c
Stores form submission data.

| Field | Type | Description |
|-------|------|-------------|
| Name | Auto Number | Response identifier |
| Form_Definition__c | Lookup | Reference to form definition |
| Response_JSON__c | Long Text Area | JSON of submitted values |
| Submitted_By__c | Lookup (User) | User who submitted |
| Submission_Date__c | DateTime | Submission timestamp |
| Target_Record_Id__c | Text | ID of created target record |

## Components

| Component | Description | Target |
|-----------|-------------|--------|
| `formBuilder` | Main drag-and-drop form builder | App Page, Tab |
| `formRenderer` | Renders forms for submission | App Page, Record Page, Community |
| `formList` | Lists available forms | App Page, Tab, Community |
| `formResponseViewer` | View and analyze responses | App Page, Tab |

## Apex Classes

| Class | Description |
|-------|-------------|
| `FormBuilderController` | Main controller for CRUD operations |
| `FormBuilderControllerTest` | Test class with 90%+ coverage |

## Field Types

| Type | Component | Description |
|------|-----------|-------------|
| text | lightning-input | Single-line text input |
| email | lightning-input | Email input with validation |
| phone | lightning-input | Phone number input |
| number | lightning-input | Numeric input |
| date | lightning-input | Date picker |
| datetime | lightning-input | Date and time picker |
| textarea | lightning-textarea | Multi-line text input |
| checkbox | lightning-input | Boolean checkbox |
| picklist | lightning-combobox | Single-select dropdown |
| radio | lightning-radio-group | Radio button group |
| multipicklist | lightning-dual-listbox | Multi-select with dual list |
| file | lightning-input | File upload |
| section | Custom | Section header |
| helptext | Custom | Help text block |

## Target Object Integration

When a target object is configured:

1. Form fields can be mapped to object fields
2. On submission, a new record is created in the target object
3. The record ID is stored in the Form Response for reference
4. Field values are automatically converted to the correct data type

## Security

- All Apex classes use `WITH SECURITY_ENFORCED` for SOQL queries
- Field-level security is respected
- Two permission sets provided for different access levels
- Sharing model set to `ReadWrite` for flexibility

## Customization

### Adding New Field Types

1. Add the field type to `inputElements`, `selectionElements`, or `layoutElements` in `formBuilder.js`
2. Add template rendering in `formBuilder.html` and `formRenderer.html`
3. Add type flags in `getDefaultFieldConfig()` method
4. Handle the field in `setFieldValue()` in `FormBuilderController.cls`

### Styling

- Customize CSS in component `.css` files
- Uses SLDS (Salesforce Lightning Design System) variables
- Responsive design with media queries

## Troubleshooting

### Form not saving
- Ensure form name is provided
- Check for Apex exceptions in browser console
- Verify permission set assignment

### Target record not created
- Verify target object API name is correct
- Check field mappings match target object fields
- Ensure user has create permission on target object

### Fields not displaying
- Check Form_JSON__c field has valid JSON
- Verify form is marked as Active
- Check browser console for JavaScript errors

## License

MIT License - feel free to use and modify for your Salesforce implementations.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Support

For issues and feature requests, please open a GitHub issue.
