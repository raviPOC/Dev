# Event Form Builder Options (Salesforce Lightning + Experience Cloud)

Below are all the practical options for letting an admin **create a form per
Event record** (custom object) with **drag-and-drop** or **click-to-add**
inputs, then display that form on an **Experience Cloud** site. Each option
includes how admins build the form, how it’s stored, and how it’s rendered.

---

## 1) **Screen Flow (no-code, drag-and-drop)**

**How admin builds the form**
- Admin uses **Flow Builder** to create a **Screen Flow** (drag-and-drop input
  components, labels, help text, validation, etc.).
- Each flow can represent a form definition.

**How it’s associated to an Event**
- Store the **Flow API Name** on the Event record (e.g., `Event_Form_Flow__c`).
- Optionally, use **Record Type** to determine which flow to render.

**How it’s rendered on Experience site**
- Use the **Flow** component in Experience Builder, or a small **LWC** wrapper
  that reads the Event record and dynamically starts the flow (`lightning-flow`)
  and passes the Event Id as an input.

**Pros**
- 100% click-based, no code for the form itself.
- Strong validation and conditional visibility.

**Cons**
- Per-event uniqueness requires a unique flow per event or a metadata-driven
  flow with dynamic components.

---

## 2) **OmniStudio OmniScript (drag-and-drop)**

**How admin builds the form**
- Admin uses **OmniScript** designer (drag-and-drop steps and fields).

**How it’s associated to an Event**
- Store OmniScript key on Event (`Type/Subtype/Language`).

**How it’s rendered**
- Embed OmniScript in Experience Cloud.

**Pros**
- Strong dynamic forms, conditional logic, multi-step.
- Great for complex guided experiences.

**Cons**
- Requires **OmniStudio license** (Salesforce Industries).

---

## 3) **Custom Form Builder (LWC + JSON schema)**

**How admin builds the form**
- Build a custom **Form Builder** Lightning App Page using LWC.
- Admin **drags and drops** or **clicks** to add input elements:
  text, textarea, picklist, checkbox, date, file upload, etc.

**How it’s stored**
- Save a **JSON schema** per Event (or per Event Form record):
  - Create `Event_Form__c` with fields like:
    - `Event__c` lookup
    - `Schema__c` (Long Text Area for JSON)
    - `Status__c` (Draft/Published)
  - Store layout, field definitions, labels, help text, validation.

**How it’s rendered**
- LWC renderer on Experience site reads `Schema__c`, generates inputs,
  saves responses to a related object (e.g., `Event_Form_Response__c`).

**Pros**
- True “free-form” per event.
- Full control over UX, validations, and storage.

**Cons**
- Requires custom build (LWC + Apex + schema design).

---

## 4) **Field Sets + Custom LWC (click-to-configure)**

**How admin builds the form**
- Admin defines fields in a **Field Set** on Event (click-based).

**How it’s stored**
- Field Set metadata references fields on Event or a related object.

**How it’s rendered**
- Use `lightning-record-edit-form` or a custom LWC that renders field sets.
- Experience site includes the LWC on the Event detail page.

**Pros**
- Click-based, no custom builder UI.
- Uses native metadata.

**Cons**
- Not fully “free-form” per record (shared per field set).
- Limited to fields already on the object.

---

## 5) **Record Types + Page Layouts (no custom code)**

**How admin builds the form**
- Use **Record Types + Page Layouts** to define different layouts.

**How it’s stored**
- Field configuration is native in layouts.

**How it’s rendered**
- Experience site record page uses **Record Detail** component.

**Pros**
- Native, very low effort.

**Cons**
- Not per record. Only per record type.
- No drag-and-drop field creation inside a single record.

---

## 6) **Salesforce Surveys (form builder + Experience embed)**

**How admin builds the form**
- Use **Salesforce Survey Builder** (drag-and-drop questions).

**How it’s stored**
- Survey metadata, with responses in Survey Response objects.

**How it’s rendered**
- Embed Survey in Experience site and tie it to Event.

**Pros**
- No-code form builder.
- Good for one-way data capture.

**Cons**
- Limited customization and theming.
- Data model differs from custom object responses.

---

## 7) **Third‑Party Form Builders (FormAssembly, Formstack, Jotform, etc.)**

**How admin builds the form**
- Use vendor form builder UI (drag/drop).

**How it’s stored**
- Vendor-side + sync to Salesforce via connector.

**How it’s rendered**
- Embed via iframe or component in Experience site.

**Pros**
- Very fast to deploy.
- Lots of UX features.

**Cons**
- Additional licensing and external dependency.
- Data residency and integration constraints.

---

## Recommended Path (based on requirements)

If the requirement is **“a unique form per Event record”** with **drag‑and‑drop
or click‑to‑add inputs**, the best options are:

1. **Custom Form Builder (LWC + JSON schema)**  
   - Most flexible and per-record.
2. **Screen Flow** with **flow-per-event** or a **metadata-driven flow**  
   - Most no-code, but less flexible per record.
3. **OmniStudio** (if licensed)  
   - Strong admin UX, drag-and-drop.

---

## Example: Minimal Custom Builder Architecture

**Objects**
- `Event__c` (existing)
- `Event_Form__c` (one per Event)
  - `Event__c` (Lookup)
  - `Schema__c` (Long Text, JSON)
  - `Status__c` (Draft/Published)
- `Event_Form_Response__c`
  - `Event__c` (Lookup)
  - `Event_Form__c` (Lookup)
  - `Response__c` (Long Text JSON)

**Components**
- `eventFormBuilder` (admin UI, drag/drop)
- `eventFormRenderer` (Experience site UI)

**Process**
1. Admin builds and saves schema.
2. Renderer reads schema and shows inputs.
3. Submissions saved to responses.

---

If you want, I can implement one of these options in code (e.g., the LWC-based
form builder with JSON schema storage and Experience site renderer).