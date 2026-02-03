import { LightningElement, track } from 'lwc';
import { ShowToastEvent } from 'lightning/platformShowToastEvent';

import listForms from '@salesforce/apex/FormBuilderController.listForms';
import getForm from '@salesforce/apex/FormBuilderController.getForm';
import saveForm from '@salesforce/apex/FormBuilderController.saveForm';

const PALETTE = [
    { type: 'text', label: 'Text' },
    { type: 'textarea', label: 'Textarea' },
    { type: 'number', label: 'Number' },
    { type: 'date', label: 'Date' },
    { type: 'checkbox', label: 'Checkbox' },
    { type: 'picklist', label: 'Picklist' }
];

function uid() {
    return `el_${Date.now()}_${Math.random().toString(16).slice(2)}`;
}

function createDefaultElement(type) {
    const base = {
        id: uid(),
        type,
        label: `${type.charAt(0).toUpperCase()}${type.slice(1)} field`,
        fieldApiName: '',
        required: false,
        placeholder: '',
        picklistValues: ''
    };
    if (type === 'checkbox') {
        base.label = 'Checkbox';
    }
    if (type === 'picklist') {
        base.picklistValues = 'Option 1\nOption 2';
    }
    return base;
}

export default class FormBuilder extends LightningElement {
    palette = PALETTE;

    @track forms = [];
    @track elements = [];

    selectedFormId = '';
    formName = '';
    targetObjectApiName = '';
    isActive = false;

    selectedIndex = null;

    dragSourceIndex = null;
    dragPaletteType = null;

    connectedCallback() {
        this.refreshForms();
    }

    get formOptions() {
        return (this.forms || []).map((f) => ({ label: f.Name, value: f.Id }));
    }

    get hasElements() {
        return (this.elements || []).length > 0;
    }

    get selectedElement() {
        if (this.selectedIndex === null) return null;
        return this.elements[this.selectedIndex] || null;
    }

    get selectedIsPicklist() {
        return this.selectedElement?.type === 'picklist';
    }

    get selectedIsTextLike() {
        const t = this.selectedElement?.type;
        return t === 'text' || t === 'textarea' || t === 'number';
    }

    get definitionJsonPretty() {
        try {
            const def = {
                version: 1,
                targetObjectApiName: this.targetObjectApiName?.trim(),
                elements: this.elements.map(({ rowClass, ...rest }) => rest)
            };
            return JSON.stringify(def, null, 2);
        } catch (e) {
            return '';
        }
    }

    refreshForms() {
        listForms()
            .then((rows) => {
                this.forms = rows || [];
            })
            .catch((err) => {
                this.toast('Error', this.normalizeError(err), 'error');
            });
    }

    handleSelectForm(event) {
        const formId = event.detail.value;
        this.selectedFormId = formId;
        if (!formId) return;

        getForm({ formId })
            .then((f) => {
                this.formName = f?.Name || '';
                this.targetObjectApiName = f?.Target_Object__c || '';
                this.isActive = !!f?.Active__c;

                let def = null;
                try {
                    def = f?.Definition__c ? JSON.parse(f.Definition__c) : null;
                } catch (e) {
                    def = null;
                }

                const els = Array.isArray(def?.elements) ? def.elements : [];
                this.selectedIndex = null;
                this.elements = this.decorateRows(els);
            })
            .catch((err) => {
                this.toast('Error', this.normalizeError(err), 'error');
            });
    }

    handleFormNameChange(event) {
        this.formName = event.target.value;
    }

    handleTargetObjectChange(event) {
        this.targetObjectApiName = event.target.value;
    }

    handleActiveChange(event) {
        this.isActive = event.target.checked;
    }

    handleNew() {
        this.selectedFormId = '';
        this.formName = '';
        this.targetObjectApiName = '';
        this.isActive = false;
        this.elements = [];
        this.selectedIndex = null;
    }

    handlePaletteDragStart(event) {
        const type = event.currentTarget.dataset.palettetype;
        this.dragPaletteType = type;
        event.dataTransfer.setData('text/plain', `palette:${type}`);
        event.dataTransfer.dropEffect = 'copy';
    }

    handleCanvasDragOver(event) {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'copy';
    }

    handleCanvasDrop(event) {
        event.preventDefault();
        const payload = event.dataTransfer.getData('text/plain') || '';
        if (payload.startsWith('palette:')) {
            const type = payload.split(':')[1];
            this.addElement(type);
        }
        this.dragPaletteType = null;
        this.dragSourceIndex = null;
    }

    handleElementDragStart(event) {
        const index = Number(event.currentTarget.dataset.index);
        this.dragSourceIndex = index;
        event.dataTransfer.setData('text/plain', `move:${index}`);
        event.dataTransfer.dropEffect = 'move';
    }

    handleElementDragOver(event) {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    }

    handleElementDrop(event) {
        event.preventDefault();
        const targetIndex = Number(event.currentTarget.dataset.index);
        const payload = event.dataTransfer.getData('text/plain') || '';

        if (payload.startsWith('palette:')) {
            const type = payload.split(':')[1];
            this.insertElementAt(type, targetIndex);
        } else if (payload.startsWith('move:')) {
            const sourceIndex = Number(payload.split(':')[1]);
            this.moveElement(sourceIndex, targetIndex);
        }

        this.dragPaletteType = null;
        this.dragSourceIndex = null;
    }

    handleSelectElement(event) {
        const index = Number(event.currentTarget.dataset.index);
        this.selectedIndex = index;
        this.elements = this.decorateRows(this.elements);
    }

    handleDeleteElement(event) {
        event.stopPropagation();
        const index = Number(event.currentTarget.dataset.index);
        const next = [...this.elements];
        next.splice(index, 1);

        this.elements = this.decorateRows(next);
        if (this.selectedIndex === index) this.selectedIndex = null;
        else if (this.selectedIndex !== null && this.selectedIndex > index) this.selectedIndex -= 1;
    }

    handleElementPropChange(event) {
        const prop = event.target.dataset.prop;
        if (!prop || this.selectedIndex === null) return;

        const next = [...this.elements];
        const current = { ...next[this.selectedIndex] };

        if (prop === 'required') current.required = event.target.checked;
        else current[prop] = event.target.value;

        next[this.selectedIndex] = current;
        this.elements = this.decorateRows(next);
    }

    addElement(type) {
        const next = [...this.elements, createDefaultElement(type)];
        this.elements = this.decorateRows(next);
    }

    insertElementAt(type, index) {
        const next = [...this.elements];
        next.splice(index, 0, createDefaultElement(type));
        this.elements = this.decorateRows(next);
    }

    moveElement(fromIndex, toIndex) {
        if (Number.isNaN(fromIndex) || Number.isNaN(toIndex)) return;
        if (fromIndex === toIndex) return;

        const next = [...this.elements];
        const [moved] = next.splice(fromIndex, 1);
        next.splice(toIndex, 0, moved);

        // keep selectedIndex aligned
        if (this.selectedIndex === fromIndex) this.selectedIndex = toIndex;
        else if (this.selectedIndex !== null) {
            const si = this.selectedIndex;
            const movedDown = fromIndex < toIndex;
            if (movedDown && si > fromIndex && si <= toIndex) this.selectedIndex = si - 1;
            if (!movedDown && si >= toIndex && si < fromIndex) this.selectedIndex = si + 1;
        }

        this.elements = this.decorateRows(next);
    }

    decorateRows(elements) {
        return (elements || []).map((el, idx) => {
            const isSelected = this.selectedIndex === idx;
            return {
                ...el,
                rowClass: `canvas-row slds-box slds-box_xx-small slds-m-bottom_x-small ${isSelected ? 'is-selected' : ''}`
            };
        });
    }

    async handleSave() {
        try {
            const def = {
                version: 1,
                targetObjectApiName: this.targetObjectApiName?.trim(),
                elements: this.elements.map(({ rowClass, ...rest }) => rest)
            };

            const id = await saveForm({
                formId: this.selectedFormId || null,
                name: this.formName,
                targetObjectApiName: this.targetObjectApiName,
                isActive: this.isActive,
                definitionJson: JSON.stringify(def)
            });

            this.selectedFormId = id;
            this.toast('Saved', 'Form saved successfully.', 'success');
            this.refreshForms();
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

