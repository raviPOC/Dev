import { LightningElement, track, api } from 'lwc';
import { ShowToastEvent } from 'lightning/platformShowToastEvent';
import saveFormDefinition from '@salesforce/apex/FormBuilderController.saveFormDefinition';
import getFormDefinition from '@salesforce/apex/FormBuilderController.getFormDefinition';

export default class FormBuilder extends LightningElement {
    @api recordId; // For editing existing forms
    
    @track formName = '';
    @track formDescription = '';
    @track formElements = [];
    @track selectedElement = null;
    @track showPreviewModal = false;
    @track isLoading = false;

    // Drag and drop state
    draggedElementType = null;
    draggedCanvasElementId = null;

    // Available elements that can be dragged to the canvas
    availableElements = [
        { type: 'text', label: 'Text Input', icon: 'utility:text' },
        { type: 'email', label: 'Email', icon: 'utility:email' },
        { type: 'number', label: 'Number', icon: 'utility:number_input' },
        { type: 'date', label: 'Date', icon: 'utility:date_input' },
        { type: 'phone', label: 'Phone', icon: 'utility:call' },
        { type: 'textarea', label: 'Text Area', icon: 'utility:textarea' },
        { type: 'checkbox', label: 'Checkbox', icon: 'utility:check' },
        { type: 'picklist', label: 'Picklist', icon: 'utility:picklist_type' },
        { type: 'radio', label: 'Radio Buttons', icon: 'utility:multi_select_checkbox' },
        { type: 'section', label: 'Section Header', icon: 'utility:section' },
        { type: 'richtext', label: 'Rich Text', icon: 'utility:richtextindent' },
        { type: 'file', label: 'File Upload', icon: 'utility:upload' }
    ];

    // Lifecycle hooks
    connectedCallback() {
        if (this.recordId) {
            this.loadExistingForm();
        }
    }

    // Load existing form for editing
    async loadExistingForm() {
        this.isLoading = true;
        try {
            const result = await getFormDefinition({ formId: this.recordId });
            if (result) {
                this.formName = result.Name;
                this.formDescription = result.Description__c || '';
                const definition = JSON.parse(result.Form_Definition__c);
                this.formElements = definition.elements.map(el => this.processLoadedElement(el));
            }
        } catch (error) {
            this.showToast('Error', 'Failed to load form: ' + this.reduceErrors(error), 'error');
        } finally {
            this.isLoading = false;
        }
    }

    // Process loaded element to add computed properties
    processLoadedElement(element) {
        return {
            ...element,
            ...this.getElementTypeFlags(element.type),
            typeLabel: this.getTypeLabel(element.type),
            containerClass: 'canvas-element slds-box slds-m-bottom_small'
        };
    }

    // Getters
    get hasFormElements() {
        return this.formElements && this.formElements.length > 0;
    }

    get formElementCount() {
        return this.formElements ? this.formElements.length : 0;
    }

    get isSaveDisabled() {
        return !this.formName || this.formElements.length === 0;
    }

    get isPreviewDisabled() {
        return this.formElements.length === 0;
    }

    get formDefinitionJson() {
        return JSON.stringify({
            name: this.formName,
            description: this.formDescription,
            elements: this.formElements.map(el => ({
                id: el.id,
                type: el.type,
                label: el.label,
                apiName: el.apiName,
                placeholder: el.placeholder,
                defaultValue: el.defaultValue,
                required: el.required,
                helpText: el.helpText,
                options: el.options,
                minLength: el.minLength,
                maxLength: el.maxLength
            }))
        });
    }

    // Property panel visibility getters
    get showPlaceholder() {
        if (!this.selectedElement) return false;
        return ['text', 'email', 'number', 'phone', 'textarea', 'picklist'].includes(this.selectedElement.type);
    }

    get showDefaultValue() {
        if (!this.selectedElement) return false;
        return ['text', 'email', 'number', 'phone', 'textarea'].includes(this.selectedElement.type);
    }

    get showOptions() {
        if (!this.selectedElement) return false;
        return ['picklist', 'radio'].includes(this.selectedElement.type);
    }

    get showRequired() {
        if (!this.selectedElement) return false;
        return this.selectedElement.type !== 'section';
    }

    get showHelpText() {
        if (!this.selectedElement) return false;
        return this.selectedElement.type !== 'section';
    }

    get showValidation() {
        if (!this.selectedElement) return false;
        return ['text', 'textarea'].includes(this.selectedElement.type);
    }

    // Event handlers
    handleFormNameChange(event) {
        this.formName = event.target.value;
    }

    handleFormDescriptionChange(event) {
        this.formDescription = event.target.value;
    }

    handleNewForm() {
        this.formName = '';
        this.formDescription = '';
        this.formElements = [];
        this.selectedElement = null;
    }

    // Drag and drop handlers for available elements
    handleDragStart(event) {
        this.draggedElementType = event.currentTarget.dataset.type;
        event.dataTransfer.effectAllowed = 'copy';
    }

    handleDragOver(event) {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'copy';
        event.currentTarget.classList.add('drag-over');
    }

    handleDragLeave(event) {
        event.currentTarget.classList.remove('drag-over');
    }

    handleDrop(event) {
        event.preventDefault();
        event.currentTarget.classList.remove('drag-over');

        if (this.draggedElementType) {
            this.addElement(this.draggedElementType);
            this.draggedElementType = null;
        }
    }

    // Canvas element drag handlers (for reordering)
    handleCanvasElementDragStart(event) {
        this.draggedCanvasElementId = event.currentTarget.dataset.id;
        event.dataTransfer.effectAllowed = 'move';
    }

    handleCanvasElementDragOver(event) {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    }

    handleCanvasElementDrop(event) {
        event.preventDefault();
        event.stopPropagation();

        const targetId = event.currentTarget.dataset.id;
        if (this.draggedCanvasElementId && this.draggedCanvasElementId !== targetId) {
            this.reorderElements(this.draggedCanvasElementId, targetId);
        }
        this.draggedCanvasElementId = null;
    }

    // Add new element to canvas
    addElement(type) {
        const elementId = 'element_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        const typeLabel = this.getTypeLabel(type);
        
        const newElement = {
            id: elementId,
            type: type,
            label: typeLabel + ' ' + (this.formElements.length + 1),
            apiName: type + '_' + (this.formElements.length + 1),
            placeholder: '',
            defaultValue: '',
            required: false,
            helpText: '',
            options: type === 'picklist' || type === 'radio' ? [
                { label: 'Option 1', value: 'option1' },
                { label: 'Option 2', value: 'option2' },
                { label: 'Option 3', value: 'option3' }
            ] : [],
            optionsText: type === 'picklist' || type === 'radio' ? 'Option 1\nOption 2\nOption 3' : '',
            minLength: null,
            maxLength: null,
            ...this.getElementTypeFlags(type),
            typeLabel: typeLabel,
            containerClass: 'canvas-element slds-box slds-m-bottom_small'
        };

        this.formElements = [...this.formElements, newElement];
        this.selectedElement = newElement;
    }

    // Get type label from type
    getTypeLabel(type) {
        const element = this.availableElements.find(el => el.type === type);
        return element ? element.label : type;
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

    // Reorder elements
    reorderElements(draggedId, targetId) {
        const elements = [...this.formElements];
        const draggedIndex = elements.findIndex(el => el.id === draggedId);
        const targetIndex = elements.findIndex(el => el.id === targetId);

        if (draggedIndex > -1 && targetIndex > -1) {
            const [draggedElement] = elements.splice(draggedIndex, 1);
            elements.splice(targetIndex, 0, draggedElement);
            this.formElements = elements;
        }
    }

    // Edit element
    handleEditElement(event) {
        const elementId = event.currentTarget.dataset.id;
        const element = this.formElements.find(el => el.id === elementId);
        if (element) {
            this.selectedElement = { ...element };
            // Highlight selected element
            this.formElements = this.formElements.map(el => ({
                ...el,
                containerClass: el.id === elementId 
                    ? 'canvas-element slds-box slds-m-bottom_small element-selected'
                    : 'canvas-element slds-box slds-m-bottom_small'
            }));
        }
    }

    // Delete element
    handleDeleteElement(event) {
        const elementId = event.currentTarget.dataset.id;
        this.formElements = this.formElements.filter(el => el.id !== elementId);
        if (this.selectedElement && this.selectedElement.id === elementId) {
            this.selectedElement = null;
        }
    }

    // Property change handlers
    handlePropertyChange(event) {
        const field = event.currentTarget.dataset.field;
        const value = event.target.value;
        
        if (this.selectedElement) {
            this.selectedElement = { ...this.selectedElement, [field]: value };
            this.updateFormElement(this.selectedElement);
        }
    }

    handleRequiredChange(event) {
        if (this.selectedElement) {
            this.selectedElement = { ...this.selectedElement, required: event.target.checked };
            this.updateFormElement(this.selectedElement);
        }
    }

    handleOptionsChange(event) {
        const optionsText = event.target.value;
        const options = optionsText.split('\n')
            .filter(opt => opt.trim())
            .map(opt => ({
                label: opt.trim(),
                value: opt.trim().toLowerCase().replace(/\s+/g, '_')
            }));

        if (this.selectedElement) {
            this.selectedElement = { 
                ...this.selectedElement, 
                optionsText: optionsText,
                options: options 
            };
            this.updateFormElement(this.selectedElement);
        }
    }

    // Update element in form
    updateFormElement(updatedElement) {
        this.formElements = this.formElements.map(el => 
            el.id === updatedElement.id ? { ...el, ...updatedElement } : el
        );
    }

    // Save form
    async handleSaveForm() {
        if (!this.validateForm()) {
            return;
        }

        this.isLoading = true;
        try {
            const formDefinition = {
                name: this.formName,
                description: this.formDescription,
                elements: this.formElements.map(el => ({
                    id: el.id,
                    type: el.type,
                    label: el.label,
                    apiName: el.apiName,
                    placeholder: el.placeholder,
                    defaultValue: el.defaultValue,
                    required: el.required,
                    helpText: el.helpText,
                    options: el.options,
                    minLength: el.minLength,
                    maxLength: el.maxLength
                }))
            };

            const result = await saveFormDefinition({
                formId: this.recordId || null,
                formName: this.formName,
                formDescription: this.formDescription,
                formDefinitionJson: JSON.stringify(formDefinition)
            });

            this.showToast('Success', 'Form saved successfully!', 'success');
            
            // Fire event for parent components
            this.dispatchEvent(new CustomEvent('formsaved', {
                detail: { formId: result }
            }));

        } catch (error) {
            this.showToast('Error', 'Failed to save form: ' + this.reduceErrors(error), 'error');
        } finally {
            this.isLoading = false;
        }
    }

    // Validate form
    validateForm() {
        if (!this.formName) {
            this.showToast('Error', 'Please enter a form name', 'error');
            return false;
        }

        if (this.formElements.length === 0) {
            this.showToast('Error', 'Please add at least one element to the form', 'error');
            return false;
        }

        // Check for duplicate API names
        const apiNames = this.formElements.map(el => el.apiName);
        const duplicates = apiNames.filter((name, index) => apiNames.indexOf(name) !== index);
        if (duplicates.length > 0) {
            this.showToast('Error', 'Duplicate API names found: ' + duplicates.join(', '), 'error');
            return false;
        }

        return true;
    }

    // Preview handlers
    handlePreviewForm() {
        this.showPreviewModal = true;
    }

    handleClosePreview() {
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
