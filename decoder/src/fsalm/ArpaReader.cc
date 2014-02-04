#include <algorithm>
#include <cstddef>
#include <cassert>
#include <string>
#include <vector>

#include "ArpaReader.hh"
#include "str.hh"

using namespace std;


namespace fsalm {

ArpaReader::ArpaReader(FILE *file)
{
    opt.throw_header_mismatch = true;
    opt.show_progress = true;
    opt.ignore_symbols.push_back("<UNK>");
    symbol_map = NULL;
    reset(file);
}

void
ArpaReader::reset(FILE *file)
{
    header.order = 0;
    header.num_ngrams_total = 0;
    header.num_ngrams.clear();
    m_current_order = -1;
    m_ngrams_read = 0;
    m_file = file;
    m_num_ignored = 0;
    end_reached = false;
}

void
ArpaReader::read_header()
{
    string line;
    vector<string> fields;

    // Read until \data\ reached
    //
    while (1) {
        if (!str::read_line(line, m_file, true))
            throw runtime_error("keyword \\data\\ not found");
        if (line == "\\data\\")
            break;
    }

    // Parse order counts
    //
    while (1) {
        if (!str::read_line(line, m_file, true))
            throw runtime_error("unexpected end of file while reading header");
        str::clean(line, " \t");
        if (line.empty())
            continue;

        if (line == "\\1-grams:") {
            m_current_order = 0;
            m_ngrams_read = 0;
            break;
        }

        if (line.substr(0, 6) != "ngram ")
            throw runtime_error("invalid line in header: " + line);
        string order_count = line.substr(6);
        fields = str::split(order_count, "=", false, 2);
        if (fields.size() != 2)
            throw runtime_error("invalid line in header: " + line);
        try {
            str::str2long(fields[0]);
            int c = str::str2long(fields[1]);
            header.num_ngrams.push_back(c);
            header.order++;
            header.num_ngrams_total += c;
        }
        catch (exception &e) {
            throw runtime_error("invalid line in header: " + line);
        }
    }
}

bool
ArpaReader::read_ngram()
{
    if (m_current_order < 0)
        throw runtime_error("read_ngram() called before read_header()");

    if (end_reached)
        return false;

    string line;
    vector<string> fields;

restart:
    while (1) {
        if (!str::read_line(line, m_file, true)) {
            if (opt.throw_header_mismatch)
                throw runtime_error("end of file before \\end\\\n");
            fprintf(stderr, "WARNING: end of file before \\end\\\n");
            line = "\\end\\";
        }
        m_recent_line = line;
        str::clean(line, " \t");
        if (line.empty())
            continue;

        if (line[0] == '\\') {

            if (m_ngrams_read != header.num_ngrams.at(m_current_order)) {
                if (opt.throw_header_mismatch)
                    throw runtime_error(
                        str::fmt(256, "expected number of %d-grams was %d, got %d\n",
                                 m_current_order + 1,
                                 header.num_ngrams[m_current_order], m_ngrams_read));
                fprintf(stderr, "WARNING: expected number of %d-grams was %d, "
                        "got %d\n", m_current_order + 1,
                        header.num_ngrams[m_current_order], m_ngrams_read);
                header.num_ngrams[m_current_order] = m_ngrams_read;
            }

            if (line == "\\end\\") {
                if (m_current_order < header.order - 1) {
                    if (opt.throw_header_mismatch)
                        throw runtime_error(
                            str::fmt(256, "\\end\\ after %d-grams, expected up to "
                                     "%d-grams\n", m_current_order + 1, header.order));
                    fprintf(stderr, "WARNING: \\end\\ after %d-grams, expected up to "
                            "%d-grams\n", m_current_order + 1, header.order);
                }
                end_reached = true;
                return false;
            }

            int order;
            int ret = sscanf(line.c_str(), "\\%d-grams:", &order);
            if (ret != 1)
                throw runtime_error("invalid keyword on line: " + line);

            assert(m_ngrams_read <= header.num_ngrams[m_current_order]);
            if (m_ngrams_read < header.num_ngrams[m_current_order]) {
                if (opt.throw_header_mismatch)
                    throw runtime_error(
                        str::fmt(256, "got %d n-grams for order %d, %d expected\n",
                                 m_ngrams_read, m_current_order + 1,
                                 header.num_ngrams[m_current_order]));
                fprintf(stderr, "WARNING: got %d n-grams for order %d, %d expected\n",
                        m_ngrams_read, m_current_order + 1,
                        header.num_ngrams[m_current_order]);
                header.num_ngrams[m_current_order] = m_ngrams_read;
            }

            if (order != m_current_order + 2)
                throw runtime_error("invalid order on line: " + line);

            m_ngrams_read = 0;
            m_current_order++;

            assert(m_current_order <= header.order);
            if (m_current_order == header.order) {
                if (opt.throw_header_mismatch)
                    throw runtime_error(str::fmt(256, "did not expect %d-grams\n",
                                                 m_current_order+1));
                fprintf(stderr, "WARNING: did not expect %d-grams\n",
                        m_current_order + 1);
                header.num_ngrams.push_back(0);
                header.order++;
            }

            return false;
        }
        else
            break;
    }

    // Parse fields of the ngram line
    //
    fields = str::split(line, " \t", true);
    if ((int)fields.size() < m_current_order + 2)
        throw runtime_error("too few fields on line: " + line);
    if ((int)fields.size() > m_current_order + 3)
        throw runtime_error("too many fields on line: " + line);
    if ((int)fields.size() == m_current_order + 3) {
        try {
            ngram.backoff = str::str2float(fields[m_current_order + 2]);
        }
        catch (exception &e) {
            throw runtime_error("invalid backoff value on line: " + line);
        }
    }
    else
        ngram.backoff = 0;

    try {
        ngram.log_prob = str::str2float(fields[0]);
    }
    catch (exception &e) {
        throw runtime_error("invalid log-probability on line: " + line);
    }

    ngram.symbols.clear();

    for (int i = 1; i < m_current_order + 2; i++)
        if (find(opt.ignore_symbols.begin(), opt.ignore_symbols.end(),
                 fields[i]) != opt.ignore_symbols.end())
        {
            m_num_ignored++;
            if (m_num_ignored < 10) {
                fputs("WARNING: ignored ngram:", stderr);
                for (int i = 1; i < m_current_order + 2; i++)
                    fprintf(stderr, " %s", fields[i].c_str());
                fputs("\n", stderr);
            }
            if (m_num_ignored == 100)
                fprintf(stderr, "WARNING: not printing more ignored ngrams\n");
            header.num_ngrams[m_current_order]--;
            goto restart;
        }

    if (m_current_order == 0)
        ngram.symbols.push_back(symbol_map->insert_new(fields[1]));
    else {
        try {
            for (int i = 1; i < m_current_order + 2; i++)
                ngram.symbols.push_back(symbol_map->index(fields[i]));
        }
        catch (string &str) {
            throw runtime_error(str::fmt(256, "invalid symbol in %d-gram: ",
                                         m_current_order + 1) + line);
        }
    }

    m_ngrams_read++;
    return true;
}

struct Compare {
    Compare(const vector<ArpaReader::Ngram> &ngrams)
        : ngrams(ngrams) { }
    bool operator()(int a, int b) const
    {
        return ngrams[a] < ngrams[b];
    }
    const vector<ArpaReader::Ngram> &ngrams;
};

bool
ArpaReader::read_order_ngrams(bool sort_ngrams)
{
    if (end_reached)
        return false;

    if (opt.show_progress)
        fprintf(stderr, "reading %d-grams...", m_current_order + 1);
    order_ngrams.clear();
    sorted_order.clear();
    while (read_ngram())
        order_ngrams.push_back(ngram);
    if (opt.show_progress)
        fprintf(stderr, "got %d...", (int)order_ngrams.size());

    if (sort_ngrams) {
        if (opt.show_progress)
            fprintf(stderr, "sorting...");
        sorted_order.resize(order_ngrams.size());
        for (int i = 0; i < (int)order_ngrams.size(); i++)
            sorted_order[i] = i;
        sort(sorted_order.begin(), sorted_order.end(),
             Compare(order_ngrams));
    }

    if (opt.show_progress)
        fprintf(stderr, "ok\n");

    return true;
}

};
