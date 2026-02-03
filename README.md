# LWC Form Builder (Drag & Drop)

This project contains a simple **Lightning Web Component (LWC) Form Builder** that lets an admin **drag & drop** form elements, map them to Salesforce **field API names**, and save the form definition. A separate **Dynamic Form** LWC renders the saved definition and **creates a record** in the configured target object.

## What’s included

- **Custom object**: `Form__c`
  - `Target_Object__c` (Text): target object API name (e.g., `Account`, `Case`, `CustomObj__c`)
  - `Definition__c` (Long Text): JSON form definition
  - `Active__c` (Checkbox)
- **Apex**: `FormBuilderController`
  - Lists forms, loads a form, saves a form definition
- **LWC**:
  - `formBuilder`: admin UI to build + save forms
  - `dynamicForm`: runtime UI to render a form + submit to create a record
- **Permission set**: `Form_Builder_Admin`

## How to use (high level)

1. Deploy the metadata in `force-app/` to your org.
2. Assign permission set `Form_Builder_Admin` to your admin user.
3. Add the `formBuilder` component to a Lightning App/Home page.
4. Create a form:
   - Set **Form Name**
   - Set **Target Object API Name**
   - Drag items from **Palette** into **Canvas**
   - Click an element and set **Field API Name** (must exist on the target object)
   - Save
5. Add the `dynamicForm` component to a page and set its `formId` to the saved `Form__c` record Id.

## Notes / limitations (current)

- The builder does not introspect object schema; mapping is done by **manual field API name** entry.
- Runtime submit uses `uiRecordApi.createRecord`, so it will respect FLS/sharing and will error if field API names are invalid.