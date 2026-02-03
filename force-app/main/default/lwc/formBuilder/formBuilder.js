import { LightningElement, track, wire } from 'lwc';
import { ShowToastEvent } from 'lightning/platformShowToastEvent';
import saveFormDefinition from '@salesforce/apex/FormBuilderController.saveFormDefinition';
import getFormDefinition from '@salesforce/apex/FormBuilderController.getFormDefinition';
import getAllFormDefinitions from '@salesforce/apex/FormBuilderController.getAllFormDefinitions';
import deleteFormDefinition from '@salesforce/apex/FormBuilderController.deleteFormDefinition';
import getAvailableObjects from '@salesforce/apex/FormBuilderController.getAvailableObjects';
import getObjectFields from '@salesforce/apex/FormBuilderController.getObjectFields';

export default class FormBuilder extends LightningElement {
    // Form metadata
    formId = null;
    @track formName = '';
    @track formDescription = '';
    @track targetObject = '';
    @track isActive = true;

    // Form fields
    @track formFields = [];
    @track selectedField = null;
    selectedFieldIndex = -1;

    // UI state
    @track isLoading = false;
    @track showLoadModal = false;
    @track showPreviewModal = false;
    @track isLoadingForms = false;
    @track isDragOver = false;
    @track existingForms = [];
    @track objectOptions = [];
    @track targetFieldOptions = [];
    @track activeSections = ['settings'];

    // Drag and drop state
    draggedElementType = null;
    draggedFieldId = null;

    // Available form elements
    inputElements = [
        { type: 'text', label: 'Text Input', icon: 'utility:text' },
        { type: 'email', label: 'Email', icon: 'utility:email' },
        { type: 'phone', label: 'Phone', icon: 'utility:call' },
        { type: 'number', label: 'Number', icon: 'utility:number_input' },
        { type: 'date', label: 'Date', icon: 'utility:date_input' },
        { type: 'datetime', label: 'Date/Time', icon: 'utility:date_time' },
        { type: 'textarea', label: 'Text Area', icon: 'utility:textarea' },
        { type: 'file', label: 'File Upload', icon: 'utility:attach' }
    ];

    selectionElements = [
        { type: 'checkbox', label: 'Checkbox', icon: 'utility:check' },
        { type: 'picklist', label: 'Picklist', icon: 'utility:picklist_type' },
        { type: 'radio', label: 'Radio Group', icon: 'utility:radio_button' },
        { type: 'multipicklist', label: 'Multi-Select', icon: 'utility:multi_select_checkbox' }
    ];

    layoutElements = [
        { type: 'section', label: 'Section Header', icon: 'utility:section' },
        { type: 'helptext', label: 'Help Text', icon: 'utility:info' }
    ];

    // Form list columns for load modal
    formListColumns = [
        { label: 'Form Name', fieldName: 'Name', type: 'text' },
        { label: 'Description', fieldName: 'Description__c', type: 'text' },
        { label: 'Active', fieldName: 'Is_Active__c', type: 'boolean' },
        { label: 'Last Modified', fieldName: 'LastModifiedDate', type: 'date' },
        {
            type: 'action',
            typeAttributes: {
                rowActions: [
                    { label: 'Load', name: 'load' },
                    { label: 'Delete', name: 'delete' }
                ]
            }
        }
    ];

    // Computed properties
    get hasFormFields() {
        return this.formFields && this.formFields.length > 0;
    }

    get hasExistingForms() {
        return this.existingForms && this.existingForms.length > 0;
    }

    get isSaveDisabled() {
        return !this.formName || this.formFields.length === 0;
    }

    get isPreviewDisabled() {
        return this.formFields.length === 0;
    }

    get formCanvasClass() {
        let baseClass = 'form-canvas slds-box slds-p-around_medium';
        if (this.isDragOver) {
            baseClass += ' drag-over';
        }
        if (!this.hasFormFields) {
            baseClass += ' empty-canvas';
        }
        return baseClass;
    }

    get hasTargetObject() {
        return !!this.targetObject;
    }

    get showPlaceholder() {
        if (!this.selectedField) return false;
        const types = ['text', 'email', 'phone', 'number', 'textarea', 'picklist'];
        return types.includes(this.selectedField.type);
    }

    get showHelpText() {
        if (!this.selectedField) return false;
        const types = ['section', 'helptext'];
        return !types.includes(this.selectedField.type);
    }

    get showOptions() {
        if (!this.selectedField) return false;
        const types = ['picklist', 'radio', 'multipicklist'];
        return types.includes(this.selectedField.type);
    }

    get showRequired() {
        if (!this.selectedField) return false;
        const types = ['section', 'helptext'];
        return !types.includes(this.selectedField.type);
    }

    get showDefaultValue() {
        if (!this.selectedField) return false;
        const types = ['text', 'email', 'phone', 'number', 'textarea'];
        return types.includes(this.selectedField.type);
    }

    get formJsonString() {
        return JSON.stringify({
            formName: this.formName,
            formDescription: this.formDescription,
            targetObject: this.targetObject,
            fields: this.formFields.map(field => ({
                id: field.id,
                type: field.type,
                label: field.label,
                apiName: field.apiName,
                placeholder: field.placeholder,
                helpText: field.helpText,
                required: field.required,
                defaultValue: field.defaultValue,
                options: field.options,
                targetField: field.targetField
            }))
        });
    }

    // Lifecycle hooks
    connectedCallback() {
        this.loadAvailableObjects();
    }

    // Load available Salesforce objects
    async loadAvailableObjects() {
        try {
            const objects = await getAvailableObjects();
            this.objectOptions = objects.map(obj => ({
                label: obj.label,
                value: obj.value
            }));
        } catch (error) {
            console.error('Error loading objects:', error);
        }
    }

    // Load fields for selected target object
    async loadTargetObjectFields() {
        if (!this.targetObject) {
            this.targetFieldOptions = [];
            return;
        }

        try {
            const fields = await getObjectFields({ objectName: this.targetObject });
            this.targetFieldOptions = fields.map(field => ({
                label: `${field.label} (${field.value})`,
                value: field.value
            }));
        } catch (error) {
            console.error('Error loading fields:', error);
            this.targetFieldOptions = [];
        }
    }

    // Event handlers - Form Settings
    handleFormNameChange(event) {
        this.formName = event.target.value;
    }

    handleFormDescriptionChange(event) {
        this.formDescription = event.target.value;
    }

    handleTargetObjectChange(event) {
        this.targetObject = event.detail.value;
        this.loadTargetObjectFields();
    }

    handleActiveChange(event) {
        this.isActive = event.target.checked;
    }

    // Drag and Drop handlers - Palette
    handleDragStart(event) {
        this.draggedElementType = event.currentTarget.dataset.type;
        this.draggedFieldId = null;
        event.dataTransfer.effectAllowed = 'copy';
    }

    // Drag and Drop handlers - Canvas
    handleDragOver(event) {
        event.preventDefault();
        event.dataTransfer.dropEffect = this.draggedFieldId ? 'move' : 'copy';
        this.isDragOver = true;
    }

    handleDragLeave(event) {
        this.isDragOver = false;
    }

    handleDrop(event) {
        event.preventDefault();
        this.isDragOver = false;

        if (this.draggedElementType) {
            // Adding new element from palette
            this.addNewField(this.draggedElementType);
            this.draggedElementType = null;
        }
    }

    // Drag and Drop handlers - Reordering fields
    handleFieldDragStart(event) {
        this.draggedFieldId = event.currentTarget.dataset.fieldId;
        this.draggedElementType = null;
        event.dataTransfer.effectAllowed = 'move';
    }

    handleFieldDragOver(event) {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    }

    handleFieldDrop(event) {
        event.preventDefault();
        event.stopPropagation();
        
        const targetFieldId = event.currentTarget.dataset.fieldId;
        
        if (this.draggedFieldId && this.draggedFieldId !== targetFieldId) {
            // Reorder fields
            const fromIndex = this.formFields.findIndex(f => f.id === this.draggedFieldId);
            const toIndex = this.formFields.findIndex(f => f.id === targetFieldId);
            
            if (fromIndex !== -1 && toIndex !== -1) {
                const [movedField] = this.formFields.splice(fromIndex, 1);
                this.formFields.splice(toIndex, 0, movedField);
                this.formFields = [...this.formFields];
                this.updateFieldContainerClasses();
            }
        } else if (this.draggedElementType) {
            // Adding new element at specific position
            const targetIndex = this.formFields.findIndex(f => f.id === targetFieldId);
            this.addNewField(this.draggedElementType, targetIndex);
            this.draggedElementType = null;
        }
        
        this.draggedFieldId = null;
        this.isDragOver = false;
    }

    // Add new field to form
    addNewField(type, atIndex = -1) {
        const fieldId = 'field_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        const fieldConfig = this.getDefaultFieldConfig(type, fieldId);
        
        if (atIndex >= 0 && atIndex < this.formFields.length) {
            this.formFields.splice(atIndex, 0, fieldConfig);
            this.formFields = [...this.formFields];
        } else {
            this.formFields = [...this.formFields, fieldConfig];
        }
        
        this.updateFieldContainerClasses();
        this.selectField(fieldId);
        
        this.showToast('Success', `Added ${fieldConfig.label} field`, 'success');
    }

    // Get default configuration for a field type
    getDefaultFieldConfig(type, fieldId) {
        const baseConfig = {
            id: fieldId,
            type: type,
            label: this.getDefaultLabel(type),
            apiName: fieldId,
            placeholder: '',
            helpText: '',
            required: false,
            defaultValue: '',
            options: [],
            optionsText: '',
            targetField: '',
            containerClass: 'form-field slds-box slds-p-around_small slds-m-bottom_small'
        };

        // Add type-specific flags
        baseConfig.isText = type === 'text';
        baseConfig.isEmail = type === 'email';
        baseConfig.isPhone = type === 'phone';
        baseConfig.isNumber = type === 'number';
        baseConfig.isDate = type === 'date';
        baseConfig.isDateTime = type === 'datetime';
        baseConfig.isTextarea = type === 'textarea';
        baseConfig.isCheckbox = type === 'checkbox';
        baseConfig.isPicklist = type === 'picklist';
        baseConfig.isRadio = type === 'radio';
        baseConfig.isMultiPicklist = type === 'multipicklist';
        baseConfig.isSection = type === 'section';
        baseConfig.isHelpText = type === 'helptext';
        baseConfig.isFile = type === 'file';

        // Default options for selection fields
        if (['picklist', 'radio', 'multipicklist'].includes(type)) {
            baseConfig.options = [
                { label: 'Option 1', value: 'option1' },
                { label: 'Option 2', value: 'option2' },
                { label: 'Option 3', value: 'option3' }
            ];
            baseConfig.optionsText = 'Option 1\nOption 2\nOption 3';
        }

        // Default help text for helptext element
        if (type === 'helptext') {
            baseConfig.helpText = 'Add helpful information here';
        }

        return baseConfig;
    }

    // Get default label for field type
    getDefaultLabel(type) {
        const labels = {
            text: 'Text Field',
            email: 'Email Address',
            phone: 'Phone Number',
            number: 'Number',
            date: 'Date',
            datetime: 'Date/Time',
            textarea: 'Text Area',
            checkbox: 'Checkbox',
            picklist: 'Picklist',
            radio: 'Radio Group',
            multipicklist: 'Multi-Select',
            section: 'Section Header',
            helptext: 'Help Text',
            file: 'File Upload'
        };
        return labels[type] || 'Field';
    }

    // Field selection and editing
    handleEditField(event) {
        const fieldId = event.currentTarget.dataset.fieldId;
        this.selectField(fieldId);
    }

    selectField(fieldId) {
        this.selectedFieldIndex = this.formFields.findIndex(f => f.id === fieldId);
        if (this.selectedFieldIndex !== -1) {
            this.selectedField = { ...this.formFields[this.selectedFieldIndex] };
            this.updateFieldContainerClasses();
        }
    }

    updateFieldContainerClasses() {
        this.formFields = this.formFields.map(field => ({
            ...field,
            containerClass: `form-field slds-box slds-p-around_small slds-m-bottom_small${
                this.selectedField && field.id === this.selectedField.id ? ' selected' : ''
            }`,
            targetFieldLabel: field.targetField ? `→ ${field.targetField}` : ''
        }));
    }

    // Property change handler
    handlePropertyChange(event) {
        const property = event.currentTarget.dataset.property;
        let value;

        if (property === 'required') {
            value = event.target.checked;
        } else {
            value = event.detail ? event.detail.value : event.target.value;
        }

        this.selectedField = { ...this.selectedField, [property]: value };

        // Handle options text conversion
        if (property === 'optionsText') {
            const lines = value.split('\n').filter(line => line.trim());
            this.selectedField.options = lines.map((line, index) => ({
                label: line.trim(),
                value: 'option' + (index + 1)
            }));
        }
    }

    // Apply property changes
    handleApplyChanges() {
        if (this.selectedFieldIndex !== -1 && this.selectedField) {
            this.formFields[this.selectedFieldIndex] = { ...this.selectedField };
            this.formFields = [...this.formFields];
            this.updateFieldContainerClasses();
            this.showToast('Success', 'Field properties updated', 'success');
        }
    }

    // Delete field
    handleDeleteField(event) {
        const fieldId = event.currentTarget.dataset.fieldId;
        const index = this.formFields.findIndex(f => f.id === fieldId);
        
        if (index !== -1) {
            this.formFields.splice(index, 1);
            this.formFields = [...this.formFields];
            
            if (this.selectedField && this.selectedField.id === fieldId) {
                this.selectedField = null;
                this.selectedFieldIndex = -1;
            }
            
            this.showToast('Success', 'Field removed', 'success');
        }
    }

    // Form actions
    handleNewForm() {
        this.formId = null;
        this.formName = '';
        this.formDescription = '';
        this.targetObject = '';
        this.isActive = true;
        this.formFields = [];
        this.selectedField = null;
        this.selectedFieldIndex = -1;
        this.targetFieldOptions = [];
    }

    handleLoadForm() {
        this.showLoadModal = true;
        this.loadExistingForms();
    }

    async loadExistingForms() {
        this.isLoadingForms = true;
        try {
            this.existingForms = await getAllFormDefinitions();
        } catch (error) {
            this.showToast('Error', 'Failed to load forms: ' + error.body?.message, 'error');
        } finally {
            this.isLoadingForms = false;
        }
    }

    handleCloseLoadModal() {
        this.showLoadModal = false;
    }

    async handleFormRowAction(event) {
        const action = event.detail.action;
        const row = event.detail.row;

        if (action.name === 'load') {
            await this.loadForm(row.Id);
            this.showLoadModal = false;
        } else if (action.name === 'delete') {
            await this.deleteForm(row.Id);
        }
    }

    async loadForm(formId) {
        this.isLoading = true;
        try {
            const form = await getFormDefinition({ formId: formId });
            
            this.formId = form.Id;
            this.formName = form.Name;
            this.formDescription = form.Description__c || '';
            this.targetObject = form.Target_Object__c || '';
            this.isActive = form.Is_Active__c;
            
            // Load target object fields if set
            if (this.targetObject) {
                await this.loadTargetObjectFields();
            }
            
            // Parse form JSON
            if (form.Form_JSON__c) {
                const formData = JSON.parse(form.Form_JSON__c);
                this.formFields = (formData.fields || []).map(field => {
                    const config = this.getDefaultFieldConfig(field.type, field.id);
                    return {
                        ...config,
                        ...field,
                        optionsText: field.options ? field.options.map(o => o.label).join('\n') : '',
                        targetFieldLabel: field.targetField ? `→ ${field.targetField}` : ''
                    };
                });
            }
            
            this.selectedField = null;
            this.selectedFieldIndex = -1;
            
            this.showToast('Success', 'Form loaded successfully', 'success');
        } catch (error) {
            this.showToast('Error', 'Failed to load form: ' + error.body?.message, 'error');
        } finally {
            this.isLoading = false;
        }
    }

    async deleteForm(formId) {
        if (!confirm('Are you sure you want to delete this form?')) {
            return;
        }

        this.isLoading = true;
        try {
            await deleteFormDefinition({ formId: formId });
            await this.loadExistingForms();
            this.showToast('Success', 'Form deleted successfully', 'success');
        } catch (error) {
            this.showToast('Error', 'Failed to delete form: ' + error.body?.message, 'error');
        } finally {
            this.isLoading = false;
        }
    }

    async handleSaveForm() {
        if (!this.formName) {
            this.showToast('Error', 'Please enter a form name', 'error');
            return;
        }

        if (this.formFields.length === 0) {
            this.showToast('Error', 'Please add at least one field to the form', 'error');
            return;
        }

        this.isLoading = true;
        try {
            const formJson = JSON.stringify({
                fields: this.formFields.map(field => ({
                    id: field.id,
                    type: field.type,
                    label: field.label,
                    apiName: field.apiName,
                    placeholder: field.placeholder,
                    helpText: field.helpText,
                    required: field.required,
                    defaultValue: field.defaultValue,
                    options: field.options,
                    targetField: field.targetField
                }))
            });

            const savedFormId = await saveFormDefinition({
                formId: this.formId,
                formName: this.formName,
                formDescription: this.formDescription,
                formJson: formJson,
                targetObject: this.targetObject,
                isActive: this.isActive
            });

            this.formId = savedFormId;
            this.showToast('Success', 'Form saved successfully', 'success');
        } catch (error) {
            this.showToast('Error', 'Failed to save form: ' + error.body?.message, 'error');
        } finally {
            this.isLoading = false;
        }
    }

    handlePreviewForm() {
        this.showPreviewModal = true;
    }

    handleClosePreviewModal() {
        this.showPreviewModal = false;
    }

    // Utility methods
    showToast(title, message, variant) {
        this.dispatchEvent(new ShowToastEvent({
            title: title,
            message: message,
            variant: variant
        }));
    }
}
