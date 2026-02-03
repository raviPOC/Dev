import { LightningElement, track, wire } from 'lwc';
import { ShowToastEvent } from 'lightning/platformShowToastEvent';
import { refreshApex } from '@salesforce/apex';
import getActiveFormDefinitions from '@salesforce/apex/FormBuilderController.getActiveFormDefinitions';
import getFormSubmissions from '@salesforce/apex/FormBuilderController.getFormSubmissions';
import getFormSubmission from '@salesforce/apex/FormBuilderController.getFormSubmission';

export default class FormSubmissionViewer extends LightningElement {
    @track selectedFormId = '';
    @track submissions = [];
    @track formOptions = [];
    @track selectedSubmission = null;
    @track showDetailModal = false;
    @track isLoading = false;
    @track submissionDataFields = [];

    wiredSubmissionsResult;

    // Table columns
    columns = [
        { label: 'Submission #', fieldName: 'Name', type: 'text' },
        { 
            label: 'Submitted Date', 
            fieldName: 'Submitted_Date__c', 
            type: 'date',
            typeAttributes: {
                year: 'numeric',
                month: 'short',
                day: '2-digit',
                hour: '2-digit',
                minute: '2-digit'
            }
        },
        { label: 'Status', fieldName: 'Status__c', type: 'text' },
        { 
            label: 'Submitted By', 
            fieldName: 'CreatedByName', 
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

    // Status options
    statusOptions = [
        { label: 'Submitted', value: 'Submitted' },
        { label: 'Under Review', value: 'Under Review' },
        { label: 'Approved', value: 'Approved' },
        { label: 'Rejected', value: 'Rejected' },
        { label: 'Processed', value: 'Processed' }
    ];

    // Wire form definitions
    @wire(getActiveFormDefinitions)
    wiredForms({ error, data }) {
        if (data) {
            this.formOptions = data.map(form => ({
                label: form.Name,
                value: form.Id
            }));
        } else if (error) {
            this.showToast('Error', 'Failed to load forms', 'error');
        }
    }

    // Getters
    get hasSubmissions() {
        return this.submissions && this.submissions.length > 0;
    }

    get formattedSubmittedDate() {
        if (this.selectedSubmission && this.selectedSubmission.Submitted_Date__c) {
            return new Date(this.selectedSubmission.Submitted_Date__c).toLocaleString();
        }
        return '';
    }

    get submittedByName() {
        if (this.selectedSubmission && this.selectedSubmission.CreatedBy) {
            return this.selectedSubmission.CreatedBy.Name;
        }
        return '';
    }

    // Event handlers
    async handleFormSelect(event) {
        this.selectedFormId = event.detail.value;
        await this.loadSubmissions();
    }

    async loadSubmissions() {
        if (!this.selectedFormId) {
            this.submissions = [];
            return;
        }

        this.isLoading = true;
        try {
            const result = await getFormSubmissions({ formDefinitionId: this.selectedFormId });
            this.submissions = result.map(sub => ({
                ...sub,
                CreatedByName: sub.CreatedBy ? sub.CreatedBy.Name : ''
            }));
        } catch (error) {
            this.showToast('Error', 'Failed to load submissions', 'error');
            this.submissions = [];
        } finally {
            this.isLoading = false;
        }
    }

    async handleRefresh() {
        await this.loadSubmissions();
        this.showToast('Success', 'Submissions refreshed', 'success');
    }

    async handleRowAction(event) {
        const action = event.detail.action;
        const row = event.detail.row;

        if (action.name === 'view') {
            await this.viewSubmissionDetails(row.Id);
        }
    }

    async viewSubmissionDetails(submissionId) {
        this.isLoading = true;
        try {
            this.selectedSubmission = await getFormSubmission({ submissionId: submissionId });
            this.parseSubmissionData();
            this.showDetailModal = true;
        } catch (error) {
            this.showToast('Error', 'Failed to load submission details', 'error');
        } finally {
            this.isLoading = false;
        }
    }

    parseSubmissionData() {
        this.submissionDataFields = [];
        if (this.selectedSubmission && this.selectedSubmission.Submission_Data__c) {
            try {
                const data = JSON.parse(this.selectedSubmission.Submission_Data__c);
                
                // Process formValues if it exists
                if (data.formValues) {
                    Object.keys(data.formValues).forEach(key => {
                        let value = data.formValues[key];
                        // Format the value for display
                        if (typeof value === 'boolean') {
                            value = value ? 'Yes' : 'No';
                        } else if (value === null || value === undefined) {
                            value = '-';
                        }
                        
                        this.submissionDataFields.push({
                            key: key,
                            label: this.formatLabel(key),
                            value: String(value)
                        });
                    });
                }

                // Add submitted timestamp if available
                if (data.submittedAt) {
                    this.submissionDataFields.push({
                        key: 'submittedAt',
                        label: 'Submission Timestamp',
                        value: new Date(data.submittedAt).toLocaleString()
                    });
                }

            } catch (error) {
                console.error('Error parsing submission data:', error);
                this.submissionDataFields = [{
                    key: 'raw',
                    label: 'Raw Data',
                    value: this.selectedSubmission.Submission_Data__c
                }];
            }
        }
    }

    formatLabel(apiName) {
        // Convert api_name to "Api Name" format
        return apiName
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
            .join(' ');
    }

    handleStatusChange(event) {
        const newStatus = event.detail.value;
        this.selectedSubmission = {
            ...this.selectedSubmission,
            Status__c: newStatus
        };
        // Note: Status update would require an additional Apex method
        this.showToast('Info', 'Status updated locally. Save functionality can be added.', 'info');
    }

    handleCloseModal() {
        this.showDetailModal = false;
        this.selectedSubmission = null;
        this.submissionDataFields = [];
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
