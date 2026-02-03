import { LightningElement, track, wire } from 'lwc';
import { ShowToastEvent } from 'lightning/platformShowToastEvent';
import getAllFormDefinitions from '@salesforce/apex/FormBuilderController.getAllFormDefinitions';
import getFormDefinition from '@salesforce/apex/FormBuilderController.getFormDefinition';
import getFormResponses from '@salesforce/apex/FormBuilderController.getFormResponses';

export default class FormResponseViewer extends LightningElement {
    @track formOptions = [];
    @track selectedFormId = '';
    @track formResponses = [];
    @track formFields = [];
    @track isLoading = false;
    @track showDetailModal = false;
    @track selectedResponse = null;
    @track selectedResponseFields = [];

    sortedBy = 'Submission_Date__c';
    sortedDirection = 'desc';

    // Datatable columns
    responseColumns = [
        { 
            label: 'Response #', 
            fieldName: 'Name', 
            type: 'text',
            sortable: true 
        },
        { 
            label: 'Submitted By', 
            fieldName: 'submittedByName', 
            type: 'text',
            sortable: true 
        },
        { 
            label: 'Submission Date', 
            fieldName: 'Submission_Date__c', 
            type: 'date',
            typeAttributes: {
                year: 'numeric',
                month: 'short',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit'
            },
            sortable: true
        },
        { 
            label: 'Target Record', 
            fieldName: 'Target_Record_Id__c', 
            type: 'text' 
        },
        {
            type: 'action',
            typeAttributes: {
                rowActions: [
                    { label: 'View Details', name: 'view' }
                ]
            }
        }
    ];

    // Load available forms
    @wire(getAllFormDefinitions)
    wiredForms({ data, error }) {
        if (data) {
            this.formOptions = data.map(form => ({
                label: form.Name,
                value: form.Id
            }));
        } else if (error) {
            this.showToast('Error', 'Failed to load forms', 'error');
        }
    }

    // Computed properties
    get hasResponses() {
        return this.formResponses && this.formResponses.length > 0;
    }

    get totalResponses() {
        return this.formResponses ? this.formResponses.length : 0;
    }

    get responsesToday() {
        if (!this.formResponses) return 0;
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        return this.formResponses.filter(r => {
            const submissionDate = new Date(r.Submission_Date__c);
            return submissionDate >= today;
        }).length;
    }

    get responsesThisWeek() {
        if (!this.formResponses) return 0;
        const weekAgo = new Date();
        weekAgo.setDate(weekAgo.getDate() - 7);
        return this.formResponses.filter(r => {
            const submissionDate = new Date(r.Submission_Date__c);
            return submissionDate >= weekAgo;
        }).length;
    }

    get formattedResponses() {
        return this.formResponses.map(response => ({
            ...response,
            submittedByName: response.Submitted_By__r ? response.Submitted_By__r.Name : 'Unknown'
        }));
    }

    get targetRecordUrl() {
        if (this.selectedResponse && this.selectedResponse.Target_Record_Id__c) {
            return `/${this.selectedResponse.Target_Record_Id__c}`;
        }
        return '#';
    }

    // Event handlers
    async handleFormSelect(event) {
        this.selectedFormId = event.detail.value;
        await this.loadFormData();
    }

    async loadFormData() {
        if (!this.selectedFormId) return;

        this.isLoading = true;

        try {
            // Load form definition to get field structure
            const formDef = await getFormDefinition({ formId: this.selectedFormId });
            if (formDef.Form_JSON__c) {
                const formData = JSON.parse(formDef.Form_JSON__c);
                this.formFields = formData.fields || [];
            }

            // Load responses
            this.formResponses = await getFormResponses({ formId: this.selectedFormId });
        } catch (error) {
            this.showToast('Error', 'Failed to load form data: ' + error.body?.message, 'error');
        } finally {
            this.isLoading = false;
        }
    }

    handleSort(event) {
        this.sortedBy = event.detail.fieldName;
        this.sortedDirection = event.detail.sortDirection;
        this.sortResponses();
    }

    sortResponses() {
        const data = [...this.formResponses];
        const key = this.sortedBy;
        const reverse = this.sortedDirection === 'asc' ? 1 : -1;

        data.sort((a, b) => {
            let valueA = a[key] || '';
            let valueB = b[key] || '';

            if (key === 'submittedByName') {
                valueA = a.Submitted_By__r ? a.Submitted_By__r.Name : '';
                valueB = b.Submitted_By__r ? b.Submitted_By__r.Name : '';
            }

            return reverse * ((valueA > valueB) - (valueB > valueA));
        });

        this.formResponses = data;
    }

    handleRowAction(event) {
        const action = event.detail.action;
        const row = event.detail.row;

        if (action.name === 'view') {
            this.viewResponseDetails(row);
        }
    }

    viewResponseDetails(response) {
        this.selectedResponse = {
            ...response,
            formattedDate: this.formatDateTime(response.Submission_Date__c)
        };

        // Parse response JSON and map to form fields
        try {
            const responseData = JSON.parse(response.Response_JSON__c || '{}');
            
            this.selectedResponseFields = this.formFields
                .filter(field => !['section', 'helptext'].includes(field.type))
                .map(field => {
                    const value = responseData[field.id];
                    const isArray = Array.isArray(value);
                    const isBoolean = typeof value === 'boolean';

                    return {
                        key: field.id,
                        label: field.label || field.id,
                        value: value,
                        displayValue: this.formatValue(value),
                        isArray: isArray,
                        arrayValue: isArray ? value : [],
                        isBoolean: isBoolean,
                        booleanIcon: value ? 'utility:check' : 'utility:close',
                        booleanClass: value ? 'slds-text-color_success' : 'slds-text-color_error',
                        booleanLabel: value ? 'Yes' : 'No'
                    };
                });
        } catch (error) {
            console.error('Error parsing response JSON:', error);
            this.selectedResponseFields = [];
        }

        this.showDetailModal = true;
    }

    formatValue(value) {
        if (value === null || value === undefined) return '—';
        if (typeof value === 'boolean') return value ? 'Yes' : 'No';
        if (Array.isArray(value)) return value.join(', ');
        return String(value);
    }

    formatDateTime(dateString) {
        if (!dateString) return '';
        const date = new Date(dateString);
        return date.toLocaleString(undefined, {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    handleCloseDetailModal() {
        this.showDetailModal = false;
        this.selectedResponse = null;
        this.selectedResponseFields = [];
    }

    showToast(title, message, variant) {
        this.dispatchEvent(new ShowToastEvent({
            title: title,
            message: message,
            variant: variant
        }));
    }
}
