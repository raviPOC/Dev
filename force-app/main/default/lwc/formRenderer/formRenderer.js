import { LightningElement, api, track } from 'lwc';
import { ShowToastEvent } from 'lightning/platformShowToastEvent';
import submitFormData from '@salesforce/apex/FormBuilderController.submitFormData';
import getFormDefinitionById from '@salesforce/apex/FormBuilderController.getFormDefinition';

export default class FormRenderer extends LightningElement {
    @api formDefinition; // JSON string of form definition
    @api formDefinitionId; // ID of saved form definition record
    @api previewMode = false; // Whether this is preview mode
    @api recordId; // For file uploads and submissions

    @track processedElements = [];
    @track formValues = {};
    @track isLoading = false;
    @track showSuccessMessage = false;

    formTitle = 'Form';
    formDescription = '';

    // Accepted file formats for upload
    acceptedFormats = ['.pdf', '.png', '.jpg', '.jpeg', '.doc', '.docx', '.xls', '.xlsx'];

    // Lifecycle hooks
    connectedCallback() {
        if (this.formDefinition) {
            this.processFormDefinition(this.formDefinition);
        } else if (this.formDefinitionId) {
            this.loadFormDefinition();
        }
    }

    // Watch for changes in formDefinition
    @api
    set formDefinitionJson(value) {
        if (value) {
            this.processFormDefinition(value);
        }
    }

    get formDefinitionJson() {
        return this.formDefinition;
    }

    // Process form definition and prepare elements
    processFormDefinition(definitionJson) {
        try {
            const definition = typeof definitionJson === 'string' 
                ? JSON.parse(definitionJson) 
                : definitionJson;

            this.formTitle = definition.name || 'Form';
            this.formDescription = definition.description || '';

            this.processedElements = (definition.elements || []).map(element => {
                return {
                    ...element,
                    value: element.defaultValue || '',
                    checked: false,
                    ...this.getElementTypeFlags(element.type)
                };
            });

            // Initialize form values
            this.formValues = {};
            this.processedElements.forEach(el => {
                this.formValues[el.apiName] = el.value || (el.type === 'checkbox' ? false : '');
            });

        } catch (error) {
            console.error('Error processing form definition:', error);
            this.showToast('Error', 'Failed to process form definition', 'error');
        }
    }

    // Load form definition from database
    async loadFormDefinition() {
        this.isLoading = true;
        try {
            const result = await getFormDefinitionById({ formId: this.formDefinitionId });
            if (result && result.Form_Definition__c) {
                this.processFormDefinition(result.Form_Definition__c);
            }
        } catch (error) {
            console.error('Error loading form definition:', error);
            this.showToast('Error', 'Failed to load form', 'error');
        } finally {
            this.isLoading = false;
        }
    }

    // Get element type flags for conditional rendering
    getElementTypeFlags(type) {
        return {
            isTextInput: type === 'text',
            isEmailInput: type === 'email',
            isNumberInput: type === 'number',
            isDateInput: type === 'date',
            isPhoneInput: type === 'phone',
            isTextarea: type === 'textarea',
            isCheckbox: type === 'checkbox',
            isPicklist: type === 'picklist',
            isRadioGroup: type === 'radio',
            isSectionHeader: type === 'section',
            isRichText: type === 'richtext',
            isFileUpload: type === 'file'
        };
    }

    // Getters
    get hasElements() {
        return this.processedElements && this.processedElements.length > 0;
    }

    get isPreviewMode() {
        return this.previewMode === true || this.previewMode === 'true';
    }

    get isSubmitDisabled() {
        return this.isLoading || this.isPreviewMode;
    }

    // Event handlers
    handleInputChange(event) {
        const elementId = event.currentTarget.dataset.id;
        const value = event.target.value;
        this.updateElementValue(elementId, value);
    }

    handleCheckboxChange(event) {
        const elementId = event.currentTarget.dataset.id;
        const checked = event.target.checked;
        this.updateElementValue(elementId, checked, true);
    }

    handleRichTextChange(event) {
        const elementId = event.currentTarget.dataset.id;
        const value = event.target.value;
        this.updateElementValue(elementId, value);
    }

    handleUploadFinished(event) {
        const elementId = event.currentTarget.dataset.id;
        const uploadedFiles = event.detail.files;
        const fileIds = uploadedFiles.map(file => file.documentId).join(',');
        this.updateElementValue(elementId, fileIds);
        this.showToast('Success', `${uploadedFiles.length} file(s) uploaded successfully`, 'success');
    }

    updateElementValue(elementId, value, isCheckbox = false) {
        this.processedElements = this.processedElements.map(el => {
            if (el.id === elementId) {
                if (isCheckbox) {
                    return { ...el, checked: value };
                }
                return { ...el, value: value };
            }
            return el;
        });

        // Update form values
        const element = this.processedElements.find(el => el.id === elementId);
        if (element) {
            this.formValues[element.apiName] = value;
        }
    }

    // Submit form
    async handleSubmit() {
        if (!this.validateForm()) {
            return;
        }

        this.isLoading = true;
        this.showSuccessMessage = false;

        try {
            const submissionData = {
                formDefinitionId: this.formDefinitionId || this.recordId,
                formValues: this.formValues,
                submittedAt: new Date().toISOString()
            };

            await submitFormData({
                formDefinitionId: submissionData.formDefinitionId,
                formDataJson: JSON.stringify(submissionData)
            });

            this.showSuccessMessage = true;
            this.showToast('Success', 'Form submitted successfully!', 'success');
            
            // Clear form after successful submission
            this.clearForm();

            // Fire event for parent components
            this.dispatchEvent(new CustomEvent('formsubmitted', {
                detail: { values: this.formValues }
            }));

        } catch (error) {
            console.error('Error submitting form:', error);
            this.showToast('Error', 'Failed to submit form: ' + this.reduceErrors(error), 'error');
        } finally {
            this.isLoading = false;
        }
    }

    // Validate form
    validateForm() {
        // Check all lightning-input elements
        const allInputs = [
            ...this.template.querySelectorAll('lightning-input'),
            ...this.template.querySelectorAll('lightning-textarea'),
            ...this.template.querySelectorAll('lightning-combobox'),
            ...this.template.querySelectorAll('lightning-radio-group')
        ];

        let isValid = true;

        allInputs.forEach(input => {
            if (!input.reportValidity()) {
                isValid = false;
            }
        });

        // Check required fields
        this.processedElements.forEach(element => {
            if (element.required && element.type !== 'section') {
                const value = this.formValues[element.apiName];
                if (!value || (typeof value === 'string' && value.trim() === '')) {
                    isValid = false;
                }
            }
        });

        if (!isValid) {
            this.showToast('Error', 'Please fill in all required fields', 'error');
        }

        return isValid;
    }

    // Clear form
    clearForm() {
        this.processedElements = this.processedElements.map(el => ({
            ...el,
            value: el.defaultValue || '',
            checked: false
        }));

        this.formValues = {};
        this.processedElements.forEach(el => {
            this.formValues[el.apiName] = el.defaultValue || '';
        });
    }

    // Utility methods
    showToast(title, message, variant) {
        this.dispatchEvent(new ShowToastEvent({
            title: title,
            message: message,
            variant: variant
        }));
    }

    reduceErrors(errors) {
        if (!Array.isArray(errors)) {
            errors = [errors];
        }

        return errors
            .filter(error => !!error)
            .map(error => {
                if (typeof error === 'string') {
                    return error;
                }
                if (error.body && error.body.message) {
                    return error.body.message;
                }
                if (error.message) {
                    return error.message;
                }
                return JSON.stringify(error);
            })
            .join(', ');
    }
}
