import { LightningElement, api, track } from 'lwc';
import { ShowToastEvent } from 'lightning/platformShowToastEvent';
import { createRecord } from 'lightning/uiRecordApi';

import getForm from '@salesforce/apex/FormBuilderController.getForm';

function normalizeLinesToOptions(linesText) {
    const lines = (linesText || '')
        .split('\n')
        .map((s) => s.trim())
        .filter((s) => !!s);
    return lines.map((v) => ({ label: v, value: v }));
}

export default class DynamicForm extends LightningElement {
    @api formId;
    @api submitLabel = 'Submit';

    @track elements = [];
    @track valuesByField = {};

    targetObjectApiName = '';
    isLoading = false;
    loadError = '';

    connectedCallback() {
        this.load();
    }

    @api
    async reload() {
        await this.load();
    }

    get isReady() {
        return !this.isLoading && !this.loadError && !!this.targetObjectApiName && (this.elements || []).length > 0;
    }

    get cardTitle() {
        return 'Form';
    }

    get renderElements() {
        const vals = this.valuesByField || {};
        return (this.elements || []).map((el) => {
            const type = el.type;
            const field = el.fieldApiName;
            const value = vals[field];

            return {
                ...el,
                value: value ?? '',
                checked: !!value,
                isText: type === 'text',
                isTextarea: type === 'textarea',
                isNumber: type === 'number',
                isDate: type === 'date',
                isCheckbox: type === 'checkbox',
                isPicklist: type === 'picklist',
                options: type === 'picklist' ? normalizeLinesToOptions(el.picklistValues) : []
            };
        });
    }

    async load() {
        if (!this.formId) {
            this.loadError = 'Missing formId.';
            return;
        }

        this.isLoading = true;
        this.loadError = '';

        try {
            const f = await getForm({ formId: this.formId });
            let def = null;
            try {
                def = f?.Definition__c ? JSON.parse(f.Definition__c) : null;
            } catch (e) {
                def = null;
            }

            const els = Array.isArray(def?.elements) ? def.elements : [];
            const targetFromDef = def?.targetObjectApiName;
            const targetFromRecord = f?.Target_Object__c;
            this.targetObjectApiName = (targetFromDef || targetFromRecord || '').trim();
            this.elements = els;
            this.valuesByField = {};

            if (!this.targetObjectApiName) {
                this.loadError = 'Form is missing target object API name.';
            }
            if (!els.length) {
                this.loadError = 'Form has no elements.';
            }
        } catch (e) {
            this.loadError = this.normalizeError(e);
        } finally {
            this.isLoading = false;
        }
    }

    handleInputChange(event) {
        const fieldApiName = event.target.dataset.field;
        const value = event.detail.value;
        this.valuesByField = { ...this.valuesByField, [fieldApiName]: value };
    }

    handleCheckboxChange(event) {
        const fieldApiName = event.target.dataset.field;
        const value = event.target.checked;
        this.valuesByField = { ...this.valuesByField, [fieldApiName]: value };
    }

    async handleSubmit() {
        try {
            const missing = (this.elements || [])
                .filter((el) => el.required)
                .filter((el) => {
                    const v = this.valuesByField?.[el.fieldApiName];
                    return v === undefined || v === null || v === '' || (el.type === 'checkbox' && v === false);
                })
                .map((el) => el.label || el.fieldApiName);

            if (missing.length) {
                this.toast('Missing required fields', missing.join(', '), 'error');
                return;
            }

            const fields = {};
            for (const el of this.elements || []) {
                const api = (el.fieldApiName || '').trim();
                if (!api) continue;
                const v = this.valuesByField?.[api];
                if (v === undefined) continue;
                fields[api] = v;
            }

            const recordInput = {
                apiName: this.targetObjectApiName,
                fields
            };

            const res = await createRecord(recordInput);
            this.toast('Success', `Record created: ${res.id}`, 'success');
            this.valuesByField = {};
        } catch (e) {
            this.toast('Error', this.normalizeError(e), 'error');
        }
    }

    toast(title, message, variant) {
        this.dispatchEvent(new ShowToastEvent({ title, message, variant }));
    }

    normalizeError(err) {
        const body = err?.body;
        if (Array.isArray(body)) return body.map((e) => e.message).join(', ');
        return body?.message || err?.message || 'Unknown error';
    }
}

