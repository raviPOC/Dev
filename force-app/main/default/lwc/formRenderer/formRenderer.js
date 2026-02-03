import { LightningElement, api, track } from 'lwc';
import { ShowToastEvent } from 'lightning/platformShowToastEvent';
import getFormDefinition from '@salesforce/apex/FormBuilderController.getFormDefinition';
import submitFormResponse from '@salesforce/apex/FormBuilderController.submitFormResponse';

export default class FormRenderer extends LightningElement {
    // Public properties
    @api formDefinitionId;
    @api formJson;
    @api previewMode = false;

    // Tracked properties
    @track renderedFields = [];
    @track formValues = {};
    @track isLoading = false;
    @track isSubmitting = false;
    @track isSubmitted = false;
    @track errorMessage = '';

    // Form metadata
    formTitle = 'Form';
    formDescription = '';
    successMessage = 'Your response has been submitted successfully.';

    // Lifecycle hooks
    connectedCallback() {
        if (this.formJson) {
            this.parseFormJson(this.formJson);
        } else if (this.formDefinitionId) {
            this.loadFormDefinition();
        }
    }

    // Load form definition from Salesforce
    async loadFormDefinition() {
        this.isLoading = true;
        this.errorMessage = '';

        try {
            const form = await getFormDefinition({ formId: this.formDefinitionId });
            
            if (!form.Is_Active__c && !this.previewMode) {
                this.errorMessage = 'This form is no longer accepting responses.';
                return;
            }

            this.formTitle = form.Name;
            this.formDescription = form.Description__c || '';
            
            if (form.Form_JSON__c) {
                this.parseFormJson(form.Form_JSON__c);
            }
        } catch (error) {
            this.errorMessage = 'Unable to load form. Please try again later.';
            console.error('Error loading form:', error);
        } finally {
            this.isLoading = false;
        }
    }

    // Parse form JSON and prepare fields for rendering
    parseFormJson(jsonString) {
        try {
            const formData = typeof jsonString === 'string' ? JSON.parse(jsonString) : jsonString;
            
            // Set form metadata if available
            if (formData.formName) {
                this.formTitle = formData.formName;
            }
            if (formData.formDescription) {
                this.formDescription = formData.formDescription;
            }

            // Process fields
            this.renderedFields = (formData.fields || []).map(field => {
                // Initialize field value
                this.formValues[field.id] = field.defaultValue || '';
                
                // Add type flags for template rendering
                return {
                    ...field,
                    value: field.defaultValue || '',
                    isText: field.type === 'text',
                    isEmail: field.type === 'email',
                    isPhone: field.type === 'phone',
                    isNumber: field.type === 'number',
                    isDate: field.type === 'date',
                    isDateTime: field.type === 'datetime',
                    isTextarea: field.type === 'textarea',
                    isCheckbox: field.type === 'checkbox',
                    isPicklist: field.type === 'picklist',
                    isRadio: field.type === 'radio',
                    isMultiPicklist: field.type === 'multipicklist',
                    isSection: field.type === 'section',
                    isHelpText: field.type === 'helptext',
                    isFile: field.type === 'file',
                    options: this.formatOptions(field.options)
                };
            });
        } catch (error) {
            this.errorMessage = 'Error parsing form structure.';
            console.error('Error parsing form JSON:', error);
        }
    }

    // Format options for picklist/radio/multiselect
    formatOptions(options) {
        if (!options || !Array.isArray(options)) {
            return [];
        }
        return options.map(opt => ({
            label: opt.label || opt,
            value: opt.value || opt
        }));
    }

    // Event Handlers
    handleFieldChange(event) {
        const fieldId = event.currentTarget.dataset.fieldId;
        const value = event.detail ? event.detail.value : event.target.value;
        this.updateFieldValue(fieldId, value);
    }

    handleCheckboxChange(event) {
        const fieldId = event.currentTarget.dataset.fieldId;
        const value = event.target.checked;
        this.updateFieldValue(fieldId, value);
    }

    handleMultiSelectChange(event) {
        const fieldId = event.currentTarget.dataset.fieldId;
        const value = event.detail.value;
        this.updateFieldValue(fieldId, value);
    }

    handleFileChange(event) {
        const fieldId = event.currentTarget.dataset.fieldId;
        const files = event.target.files;
        if (files && files.length > 0) {
            // For now, just store file name. Full file upload would require ContentVersion
            this.updateFieldValue(fieldId, files[0].name);
        }
    }

    updateFieldValue(fieldId, value) {
        this.formValues[fieldId] = value;
        
        // Update rendered fields for two-way binding
        this.renderedFields = this.renderedFields.map(field => {
            if (field.id === fieldId) {
                return { ...field, value: value };
            }
            return field;
        });
    }

    // Validate form
    validateForm() {
        let isValid = true;
        const inputComponents = this.template.querySelectorAll(
            'lightning-input, lightning-textarea, lightning-combobox, lightning-radio-group, lightning-dual-listbox'
        );

        inputComponents.forEach(component => {
            // Check validity if the component has reportValidity method
            if (component.reportValidity) {
                if (!component.reportValidity()) {
                    isValid = false;
                }
            }
        });

        return isValid;
    }

    // Submit form
    async handleSubmit() {
        if (this.previewMode) {
            return;
        }

        if (!this.validateForm()) {
            this.showToast('Error', 'Please fill in all required fields correctly.', 'error');
            return;
        }

        this.isSubmitting = true;

        try {
            // Filter out section and helptext fields from submission
            const submissionData = {};
            this.renderedFields.forEach(field => {
                if (!['section', 'helptext'].includes(field.type)) {
                    submissionData[field.id] = this.formValues[field.id];
                }
            });

            await submitFormResponse({
                formId: this.formDefinitionId,
                responseJson: JSON.stringify(submissionData)
            });

            this.isSubmitted = true;
            this.showToast('Success', 'Form submitted successfully!', 'success');

            // Dispatch custom event for parent components
            this.dispatchEvent(new CustomEvent('formsubmit', {
                detail: {
                    formId: this.formDefinitionId,
                    values: submissionData
                }
            }));
        } catch (error) {
            this.showToast('Error', 'Failed to submit form: ' + (error.body?.message || error.message), 'error');
            console.error('Error submitting form:', error);
        } finally {
            this.isSubmitting = false;
        }
    }

    // Reset form
    handleReset() {
        // Reset all field values
        this.renderedFields = this.renderedFields.map(field => {
            const defaultValue = field.defaultValue || '';
            this.formValues[field.id] = defaultValue;
            return { ...field, value: defaultValue };
        });

        // Clear validation errors
        const inputComponents = this.template.querySelectorAll(
            'lightning-input, lightning-textarea, lightning-combobox, lightning-radio-group, lightning-dual-listbox'
        );
        inputComponents.forEach(component => {
            if (component.setCustomValidity) {
                component.setCustomValidity('');
            }
        });

        this.showToast('Info', 'Form has been reset.', 'info');
    }

    // Submit another response
    handleSubmitAnother() {
        this.isSubmitted = false;
        this.handleReset();
    }

    // Toast notification
    showToast(title, message, variant) {
        this.dispatchEvent(new ShowToastEvent({
            title: title,
            message: message,
            variant: variant
        }));
    }

    // Public method to get form values
    @api
    getFormValues() {
        return { ...this.formValues };
    }

    // Public method to set form values
    @api
    setFormValues(values) {
        if (values && typeof values === 'object') {
            Object.keys(values).forEach(key => {
                this.updateFieldValue(key, values[key]);
            });
        }
    }

    // Public method to validate
    @api
    validate() {
        return this.validateForm();
    }
}
