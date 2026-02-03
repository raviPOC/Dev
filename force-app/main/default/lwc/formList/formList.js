import { LightningElement, track, wire } from 'lwc';
import { ShowToastEvent } from 'lightning/platformShowToastEvent';
import { refreshApex } from '@salesforce/apex';
import getActiveFormDefinitions from '@salesforce/apex/FormBuilderController.getActiveFormDefinitions';

export default class FormList extends LightningElement {
    @track activeForms = [];
    @track isLoading = true;
    @track showFormModal = false;
    @track selectedFormId = null;
    @track selectedFormName = '';

    wiredFormsResult;

    @wire(getActiveFormDefinitions)
    wiredForms(result) {
        this.wiredFormsResult = result;
        const { data, error } = result;
        
        if (data) {
            this.activeForms = data.map(form => ({
                ...form,
                formattedDate: this.formatDate(form.LastModifiedDate)
            }));
            this.isLoading = false;
        } else if (error) {
            this.showToast('Error', 'Failed to load forms', 'error');
            this.isLoading = false;
        }
    }

    get hasForms() {
        return this.activeForms && this.activeForms.length > 0;
    }

    formatDate(dateString) {
        if (!dateString) return '';
        const date = new Date(dateString);
        return date.toLocaleDateString(undefined, {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    }

    handleRefresh() {
        this.isLoading = true;
        refreshApex(this.wiredFormsResult).then(() => {
            this.isLoading = false;
        });
    }

    handleOpenForm(event) {
        const formId = event.currentTarget.dataset.formId;
        const form = this.activeForms.find(f => f.Id === formId);
        
        if (form) {
            this.selectedFormId = formId;
            this.selectedFormName = form.Name;
            this.showFormModal = true;
        }
    }

    handleCloseModal() {
        this.showFormModal = false;
        this.selectedFormId = null;
        this.selectedFormName = '';
    }

    handleFormSubmit(event) {
        this.showToast('Success', 'Form submitted successfully!', 'success');
        // Optionally close the modal after successful submission
        // this.handleCloseModal();
    }

    showToast(title, message, variant) {
        this.dispatchEvent(new ShowToastEvent({
            title: title,
            message: message,
            variant: variant
        }));
    }
}
